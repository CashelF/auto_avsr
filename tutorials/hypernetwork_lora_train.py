"""Train a hypernetwork that predicts LoRA adapters from video prefixes."""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

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


class HyperLoRALinear(nn.Module):
    """Linear layer with dynamic LoRA weights injected per batch."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, name: str = ""):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A_weight: Optional[torch.Tensor] = None
        self.lora_B_weight: Optional[torch.Tensor] = None

        # debug
        self.layer_name = name

    def set_adapter(self, lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
        self.lora_A_weight = lora_A
        self.lora_B_weight = lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base output
        base = F.linear(x, self.weight, self.bias)

        # Don't apply LoRA if adapters unset or we're in a no_grad region (e.g. prefix encoding) (or it will use prev lora adapter)
        if (
            self.lora_A_weight is None
            or self.lora_B_weight is None
            or not torch.is_grad_enabled()
        ):
            return base

        x_drop = self.lora_dropout(x)

        # Number of per-sample adapters
        B = self.lora_A_weight.shape[0]

        total_elems = x_drop.numel()
        per_adapter = self.in_features * B

        # If we can't evenly split x into B chunks of in_features, we don't know
        # how to align adapters with activations -> skip LoRA for this call.
        if total_elems % per_adapter != 0:
            return base

        S = total_elems // per_adapter
        x_flat = x_drop.view(B, S, self.in_features)  # (B, S, in_features)

        # lora_A: (B, rank, in_features)
        # lora_B: (B, out_features, rank)
        lora_mid = torch.einsum("bri,bsi->bsr", self.lora_A_weight, x_flat)
        lora_out = torch.einsum("bor,bsr->bso", self.lora_B_weight, lora_mid) * self.scaling

        lora_out = lora_out.reshape(*x_drop.shape[:-1], self.out_features)

        return base + lora_out




@dataclass
class AdapterSpec:
    name: str
    in_features: int
    out_features: int
    rank: int


@dataclass
class HyperLoRAConfig:
    root_dir: str
    train_file: str
    val_file: str
    pretrained_model_path: str
    modality: str = "video"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    prefix_frames: int = 4
    conditioning_dim: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    batch_size: Optional[int] = None
    train_num_buckets: int = 200
    devices: int = 1
    seed: int = 42
    max_frames: int = 1600


class EncoderConditioning(nn.Module):
    """Pool encoder outputs and project to a conditioning vector."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = make_non_pad_mask(lengths).to(encoded.device)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.project(pooled)


class LoRAHyperNetwork(nn.Module):
    """Predict LoRA adapter weights from a conditioning vector."""

    def __init__(self, specs: List[AdapterSpec], rank: int, hidden_size: int):
        super().__init__()
        self.specs = specs
        self.rank = rank
        total_params = sum((spec.in_features + spec.out_features) * rank for spec in specs)
        self.generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, total_params),
        )

    def forward(self, conditioning: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        weights = self.generator(conditioning)
        adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        idx = 0
        for spec in self.specs:
            a_size = spec.rank * spec.in_features
            b_size = spec.out_features * self.rank
            lora_A = weights[:, idx : idx + a_size].view(-1, self.rank, spec.in_features)
            idx += a_size
            lora_B = weights[:, idx : idx + b_size].view(-1, spec.out_features, self.rank)
            idx += b_size
            adapters[spec.name] = (lora_A, lora_B)
        return adapters

def should_adapt_layer(full_name: str, linear: nn.Linear) -> bool:
    # 1) Only encoder blocks
    if not full_name.startswith("encoder.encoders."):
        return False

    # 2) Skip positional projection
    if "linear_pos" in full_name:
        return False

    # Only last few encoder layers
    # if not any(f"encoder.encoders.{i}." in full_name for i in range(8, 12)):
    #     return False

    return True


def replace_linear_with_hyper_lora(
    module: nn.Module, rank: int, alpha: float, dropout: float
) -> Tuple[List[AdapterSpec], Dict[str, HyperLoRALinear]]:
    specs: List[AdapterSpec] = []
    adapters: Dict[str, HyperLoRALinear] = {}

    def _recurse(submodule: nn.Module, prefix: str = "") -> None:
        for name, child in list(submodule.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and should_adapt_layer(full_name, child):
                adapter = HyperLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, name=full_name)
                setattr(submodule, name, adapter)
                specs.append(AdapterSpec(full_name, child.in_features, child.out_features, rank))
                adapters[full_name] = adapter
            else:
                _recurse(child, full_name)

    _recurse(module)
    return specs, adapters


class HyperLoRALightningModule(LightningModule):
    def __init__(self, config: HyperLoRAConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = load_pretrained_model(config.pretrained_model_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.adapter_specs, self.adapters = replace_linear_with_hyper_lora(
            self.model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout
        )

        self.conditioning_encoder = EncoderConditioning(
            self.model.proj_encoder.out_features, config.conditioning_dim
        )
        self.hypernetwork = LoRAHyperNetwork(self.adapter_specs, config.lora_rank, config.conditioning_dim)

    def encode_batch(self, batch):
        inputs, lengths = batch["inputs"], batch["input_lengths"]
        if self.model.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(lengths).to(inputs.device).unsqueeze(-2)
        feats = self.model.frontend(inputs)
        feats = self.model.proj_encoder(feats)
        encoded, _ = self.model.encoder(feats, padding_mask)
        return encoded, lengths

    def apply_hyper_adapters(self, batch) -> None:
        inputs, lengths = batch["inputs"], batch["input_lengths"]
        prefix_frames = min(self.config.prefix_frames, inputs.shape[1])

        prefix_lengths = torch.minimum(lengths, torch.full_like(lengths, prefix_frames))
        prefix_inputs = inputs[:, :prefix_frames]
        if self.model.modality == "audio":
            prefix_lengths = torch.div(prefix_lengths, 640, rounding_mode="trunc")

        padding_mask = make_non_pad_mask(prefix_lengths).to(inputs.device).unsqueeze(-2)
        with torch.no_grad():
            feats = self.model.frontend(prefix_inputs)
            feats = self.model.proj_encoder(feats)
            encoded_prefix, _ = self.model.encoder(feats, padding_mask)

        conditioning = self.conditioning_encoder(encoded_prefix.detach(), prefix_lengths)
        predicted = self.hypernetwork(conditioning)
        for name, adapter in self.adapters.items():
            lora_A, lora_B = predicted[name]
            adapter.set_adapter(lora_A, lora_B)

    def forward(self, batch):
        self.apply_hyper_adapters(batch)
        loss, loss_ctc, loss_att, acc = self.model(
            batch["inputs"], batch["input_lengths"], batch["targets"]
        )
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

        random_batch = random.randint(0, len(batch)-1)
        if batch_idx == random_batch:
            self.print(f"Ref: {ref_texts[random_batch]}")
            self.print(f"Hyp: {hyp_texts[random_batch]}")

    def configure_optimizers(self):
        params = list(self.conditioning_encoder.parameters()) + list(self.hypernetwork.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer


class HyperLoRADataModule(DataModule):
    def __init__(self, config: HyperLoRAConfig):
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
    parser = argparse.ArgumentParser(description="Train a hypernetwork to predict LoRA adapters")
    parser.add_argument("--root-dir", required=True, help="Root directory of the preprocessed dataset")
    parser.add_argument("--train-file", required=True, help="Training label list file")
    parser.add_argument("--val-file", required=True, help="Validation label list file")
    parser.add_argument("--pretrained-model-path", required=True, help="Checkpoint to start from")
    parser.add_argument("--modality", default="video", choices=["video", "audio"], help="Input modality")
    parser.add_argument("--lora-rank", type=int, default=8, help="Low-rank dimension for LoRA adapters")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="Scaling factor for LoRA adapters")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout applied inside LoRA adapters")
    parser.add_argument("--prefix-frames", type=int, default=4, help="Number of initial frames used for conditioning")
    parser.add_argument("--conditioning-dim", type=int, default=512, help="Hidden dimension inside the hypernetwork")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the hypernetwork")
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

    config = HyperLoRAConfig(
        root_dir=args.root_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        pretrained_model_path=args.pretrained_model_path,
        modality=args.modality,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        prefix_frames=args.prefix_frames,
        conditioning_dim=args.conditioning_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_num_buckets=args.train_num_buckets,
        devices=args.devices,
        seed=args.seed,
        max_frames=args.max_frames,
    )

    datamodule = HyperLoRADataModule(config)
    module = HyperLoRALightningModule(config)

    checkpoint_cb = ModelCheckpoint(save_last=True, monitor="val_cer", mode="min")
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices if torch.cuda.is_available() else None,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10,
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

# python -m tutorials.hypernetwork_lora_train --root-dir lrs2 --train-file preprocessed_train/labels/lrs2_train_transcript_lengths_seg16s.csv --val-file preprocessed_val/labels/lrs2_val_transcript_lengths_seg16s.csv --pretrained-model-path vsr_trlrs3vox2_base.pth --modality video