"""Train LoRA adapters on a full dataset using the same injection scheme as the hypernetwork script."""

import argparse
from dataclasses import dataclass
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodule.data_module import DataModule
from datamodule.transforms import TextTransform
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from tutorials.lora_overfit import (
    character_error_rate,
    greedy_ctc_decode,
    load_pretrained_model,
    word_error_rate,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class LoRALinear(nn.Module):
    """LoRA adapter injected into a frozen linear layer."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return base
        lora = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + lora


def should_adapt_layer(full_name: str, linear: nn.Linear) -> bool:
    # Only encoder blocks
    if not full_name.startswith("encoder.encoders."):
        return False

    # Skip positional projection
    if "linear_pos" in full_name:
        return False
    
    # Only self attn
    # if 'self_attn' not in full_name:
    #     return False

    # Only last few encoder layers
    if not any(f"encoder.encoders.{i}." in full_name for i in range(10, 12)):
        return False
    
    # Only k and v vectors
    if not any(key in full_name for key in ["linear_k", "linear_v"]):
        return False

    return True


def replace_linear_with_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> List[str]:
    """Recursively inject LoRA adapters into encoder Linear layers."""

    adapted: List[str] = []

    def _recurse(submodule: nn.Module, prefix: str = "") -> None:
        for name, child in list(submodule.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and should_adapt_layer(full_name, child):
                setattr(submodule, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                adapted.append(full_name)
            else:
                _recurse(child, full_name)

    _recurse(module)
    return adapted


def freeze_non_lora_parameters(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


@dataclass
class LoRAConfig:
    root_dir: str
    train_file: str
    val_file: str
    pretrained_model_path: str
    modality: str = "video"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    batch_size: Optional[int] = None
    train_num_buckets: int = 200
    devices: int = 1
    seed: int = 42
    max_frames: int = 1600


class LoRALightningModule(LightningModule):
    def __init__(self, config: LoRAConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = load_pretrained_model(config.pretrained_model_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.adapted_layers = replace_linear_with_lora(
            self.model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout
        )
        freeze_non_lora_parameters(self.model)

        self.enable_lora = True
        self.set_lora_enabled(self.enable_lora)

    def set_lora_enabled(self, enabled: bool) -> None:
        """Enable/disable all LoRA adapters in the wrapped model."""
        self.enable_lora = enabled
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.enabled = enabled

    def encode_batch(self, batch):
        inputs, lengths = batch["inputs"], batch["input_lengths"]
        if self.model.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(lengths).to(inputs.device).unsqueeze(-2)
        feats = self.model.frontend(inputs)
        feats = self.model.proj_encoder(feats)
        encoded, _ = self.model.encoder(feats, padding_mask)
        return encoded, lengths

    def forward(self, batch):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        return loss, loss_ctc, loss_att, acc

    def training_step(self, batch, batch_idx):
        loss, loss_ctc, loss_att, acc = self.forward(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True)
        self.log("loss_att", loss_att, on_step=False, on_epoch=True)
        self.log("decoder_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ctc, loss_att, acc = self.forward(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_loss_ctc", loss_ctc, on_step=False, on_epoch=True)
        self.log("val_loss_att", loss_att, on_step=False, on_epoch=True)
        self.log("val_decoder_acc", acc, on_step=False, on_epoch=True)

        encoded, _ = self.encode_batch(batch)
        log_probs = self.model.ctc.log_softmax(encoded)
        pred_ids = greedy_ctc_decode(log_probs, blank_id=self.model.blank)

        ref_texts = [self.text_transform.post_process(t.cpu()) for t in batch["targets"]]
        hyp_texts = [self.text_transform.post_process(torch.tensor(pred, dtype=torch.long)) for pred in pred_ids]

        cer_scores = [character_error_rate(r, h) for r, h in zip(ref_texts, hyp_texts)]
        wer_scores = [word_error_rate(r, h) for r, h in zip(ref_texts, hyp_texts)]

        cer_tensor = torch.tensor(cer_scores, device=loss.device).mean()
        wer_tensor = torch.tensor(wer_scores, device=loss.device).mean()

        self.log("val_cer", cer_tensor, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_wer", wer_tensor, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.print(f"Ref: {ref_texts[0]}")
            self.print(f"Hyp: {hyp_texts[0]}")

    def configure_optimizers(self):
        # Only train LoRA parameters
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        warmup_epochs = 3
        total_epochs = self.config.max_epochs

        def lr_lambda(epoch):
            # epoch is 0-indexed
            if epoch < warmup_epochs:
                # linear warmup 0 -> 1
                return float(epoch + 1) / float(warmup_epochs)
            # cosine decay 1 -> 0
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }



class LoRADataModule(DataModule):
    def __init__(self, config: LoRAConfig):
        args = argparse.Namespace(
            root_dir=config.root_dir,
            train_file=config.train_file,
            val_file=config.val_file,
            test_file=config.val_file,
            modality=config.modality,
            max_frames=config.max_frames,
        )
        super().__init__(args=args, batch_size=config.batch_size, train_num_buckets=config.train_num_buckets)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters on a full dataset")
    parser.add_argument("--root-dir", required=True, help="Root directory of the preprocessed dataset")
    parser.add_argument("--train-file", required=True, help="Training label list file")
    parser.add_argument("--val-file", required=True, help="Validation label list file")
    parser.add_argument("--pretrained-model-path", required=True, help="Checkpoint to start from")
    parser.add_argument("--modality", default="video", choices=["video", "audio"], help="Input modality")
    parser.add_argument("--lora-rank", type=int, default=8, help="Low-rank dimension for LoRA adapters")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="Scaling factor for LoRA adapters")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout applied inside LoRA adapters")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LoRA parameters")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay applied to the optimizer")
    parser.add_argument("--max-epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (defaults to bucketing only)")
    parser.add_argument("--train-num-buckets", type=int, default=200, help="Number of buckets used for length-aware batching")
    parser.add_argument("--max-frames", type=int, default=1600, help="Maximum frames per batch")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    config = LoRAConfig(
        root_dir=args.root_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        pretrained_model_path=args.pretrained_model_path,
        modality=args.modality,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_num_buckets=args.train_num_buckets,
        devices=args.devices,
        seed=args.seed,
        max_frames=args.max_frames,
    )

    datamodule = LoRADataModule(config)
    module = LoRALightningModule(config)

    checkpoint_cb = ModelCheckpoint(save_last=True, monitor="val_cer", mode="min")
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices if torch.cuda.is_available() else None,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )

    module.set_lora_enabled(False)
    print("\n==== Initial validation before training ====")
    trainer.validate(module, datamodule=datamodule)
    print("============================================\n")
    module.set_lora_enabled(True)

    trainer.fit(module, datamodule=datamodule)
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

# python -m tutorials.lora_train --root-dir lrs2 --train-file preprocessed_train/labels/lrs2_train_transcript_lengths_seg16s.csv --val-file preprocessed_val/labels/lrs2_val_transcript_lengths_seg16s.csv --pretrained-model-path vsr_trlrs3vox2_base.pth --modality video
# Best wer: 0.509 version_150