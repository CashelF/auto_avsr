"""LoRA fine-tuning utility for overfitting a single video example."""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Dataset

from datamodule.av_dataset import load_video
from datamodule.data_module import collate_pad
from datamodule.transforms import TextTransform, VideoTransform
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


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

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + lora


def add_lora_adapters(module: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
    """Recursively replace Linear layers with LoRALinear adapters."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            add_lora_adapters(child, rank=rank, alpha=alpha, dropout=dropout)


def freeze_non_lora_parameters(module: nn.Module):
    """Freeze all parameters except LoRA adapters."""

    for name, param in module.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def _levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Compute Levenshtein distance between two token lists."""

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[-1][-1]


def character_error_rate(ref: str, hyp: str) -> float:
    ref_chars, hyp_chars = list(ref), list(hyp)
    if len(ref_chars) == 0:
        return float("inf") if len(hyp_chars) > 0 else 0.0
    return _levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words, hyp_words = ref.split(), hyp.split()
    if len(ref_words) == 0:
        return float("inf") if len(hyp_words) > 0 else 0.0
    return _levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def greedy_ctc_decode(log_probs: torch.Tensor, blank_id: int) -> List[List[int]]:
    """Collapse repeats and blanks for greedy CTC decoding."""

    pred_tokens = torch.argmax(log_probs, dim=-1)  # (B, T)
    results: List[List[int]] = []
    for seq in pred_tokens:
        collapsed = []
        prev = None
        for token in seq.tolist():
            if token == blank_id:
                prev = token
                continue
            if token != prev:
                collapsed.append(token)
            prev = token
        results.append(collapsed)
    return results


class SingleVideoDataset(Dataset):
    """Dataset that wraps a single video + transcript pair."""

    def __init__(self, video_path: str, label_text: str, video_transform: Optional[VideoTransform]):
        self.video = load_video(video_path)
        self.video = video_transform(self.video) if video_transform else self.video
        self.text_transform = TextTransform()
        self.target = self.text_transform.tokenize(label_text)

        unk_id = int(self.text_transform.hashmap["<unk>"])
        unk_count = (self.target == unk_id).sum().item()
        if unk_count:
            total = len(self.target)
            ratio = unk_count / total
            print(
                f"[SingleVideoDataset] {unk_count}/{total} tokens ({ratio:.1%}) "
                "in the provided label were <unk>. The SentencePiece vocabulary may not "
                "cover digits or special symbols in the transcript, which will appear as <unk> in refs/hyps."
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        del idx
        return {"input": self.video, "target": self.target}


class SingleVideoDataModule(LightningDataModule):
    def __init__(self, video_path: str, label_text: str):
        super().__init__()
        self.video_path = video_path
        self.label_text = label_text
        self.video_transform = VideoTransform("train")

    def train_dataloader(self):
        dataset = SingleVideoDataset(self.video_path, self.label_text, self.video_transform)
        return DataLoader(dataset, batch_size=1, collate_fn=collate_pad)

    def val_dataloader(self):
        # Overfit mode: validate on the same example
        return self.train_dataloader()


@dataclass
class LoRAConfig:
    pretrained_model_path: str
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lr: float = 1e-3
    max_epochs: int = 50
    seed: int = 42


def load_pretrained_model(path: str) -> E2E:
    model = E2E(len(TextTransform().token_list), "video")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


class LoRAOverfitModule(LightningModule):
    def __init__(self, config: LoRAConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = load_pretrained_model(config.pretrained_model_path)

        add_lora_adapters(self.model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
        freeze_non_lora_parameters(self.model)

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
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        return optimizer


def _read_label_text(label_path: str) -> str:
    """Load transcript text from a file, stripping trailing newlines."""

    with open(label_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA overfit on a single video example")
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the video file (expects a paired .wav in the same folder)",
    )
    parser.add_argument(
        "--label-path",
        required=True,
        help="Path to a text file containing the transcript for the video",
    )
    parser.add_argument("--pretrained-model-path", required=True, help="Checkpoint to start from")
    parser.add_argument("--lora-rank", type=int, default=8, help="Low-rank dimension for LoRA adapters")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="Scaling factor for LoRA adapters")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout applied inside LoRA adapters")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LoRA parameters")
    parser.add_argument("--max-epochs", type=int, default=50, help="Number of epochs to overfit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use (defaults to CPU when unavailable)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    label_text = _read_label_text(args.label_path)

    config = LoRAConfig(
        pretrained_model_path=args.pretrained_model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    module = LoRAOverfitModule(config)
    datamodule = SingleVideoDataModule(args.video_path, label_text)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else None,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=1,
        overfit_batches=1.0,
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

# python -m tutorials.lora_overfit --video-path lrs2/dataset/lrs2/lrs2_video_seg16s/main/5535415699068794046/00001.mp4 --label-path lrs2/dataset/lrs2/lrs2_text_seg16s/main/5535415699068794046/00001.txt --pretrained-model-path vsr_trlrs3vox2_base.pth --devices 1