"""Render predicted transcripts as captions on a video using a hypernetwork LoRA checkpoint."""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from datamodule.transforms import TextTransform, VideoTransform
from preparation.data.data_module import AVSRDataLoader
from tutorials.hypernetwork_lora_train import HyperLoRALightningModule, greedy_ctc_decode


def _prepare_input(
    video_path: str, device: torch.device, detector: str = "retinaface"
) -> Dict[str, torch.Tensor]:
    """Preprocess an unprocessed video and transform it into the model input format."""

    video_loader = AVSRDataLoader(
        modality="video", detector=detector, convert_gray=False, gpu_type=device.type
    )

    processed = video_loader.load_data(video_path)
    if processed is None:
        raise RuntimeError("Video preprocessing failed; no frames returned")

    video = torch.tensor(processed, dtype=torch.float32)
    if video.dim() == 3:
        video = video.unsqueeze(1)  # (T, 1, H, W)
    elif video.dim() == 4:
        if video.shape[-1] in (1, 3):
            video = video.permute(0, 3, 1, 2)  # (T, C, H, W) from HWC
    else:
        raise ValueError(f"Unexpected video shape from preprocessing: {video.shape}")

    video_transform = VideoTransform("test")
    transformed_video = video_transform(video).unsqueeze(0)  # (1, T, 1, H, W)

    batch = {
        "inputs": transformed_video.to(device),
        "input_lengths": torch.tensor([transformed_video.shape[1]], device=device),
    }
    return batch


def _predict_transcript(
    module: HyperLoRALightningModule, batch: Dict[str, torch.Tensor]
) -> Tuple[str, Dict[str, Dict[str, torch.Tensor]]]:
    """Run hypernetwork-driven inference and return the transcript and adapters."""

    module.enable_lora = True
    with torch.no_grad():
        module.apply_hyper_adapters(batch)
        encoded, _ = module.encode_batch(batch)
        log_probs = module.model.ctc.log_softmax(encoded)
        pred_ids = greedy_ctc_decode(log_probs, blank_id=module.model.blank)[0]

    text_transform = TextTransform()
    transcript = text_transform.post_process(torch.tensor(pred_ids, dtype=torch.long))

    adapters: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, adapter in module.adapters.items():
        if adapter.lora_A_weight is None or adapter.lora_B_weight is None:
            continue
        adapters[name] = {
            "lora_A": adapter.lora_A_weight.detach().cpu().clone(),
            "lora_B": adapter.lora_B_weight.detach().cpu().clone(),
        }
    return transcript, adapters


def _draw_caption(frames: torch.Tensor, text: str) -> torch.Tensor:
    """Overlay caption text onto frames (T, H, W, C) and return a tensor."""

    caption = text if text else "[no transcript]"
    font = ImageFont.load_default()
    captioned_frames = []
    for frame in frames:
        image = Image.fromarray(frame.numpy()).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 6
        x = max(0, (image.width - text_w) // 2)
        y = max(0, image.height - text_h - 2 * padding)
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=(0, 0, 0, 180),
        )
        draw.text((x, y), caption, font=font, fill=(255, 255, 255, 255))
        composed = Image.alpha_composite(image, overlay).convert("RGB")
        captioned_frames.append(torch.from_numpy(np.array(composed)))
    return torch.stack(captioned_frames)


def render_captions(
    video_path: str,
    hyper_ckpt: str,
    output_path: str,
    device: torch.device,
    detector: str = "retinaface",
    save_adapter_path: str = "",
) -> str:
    """Generate captions with the hypernetwork and write them onto the video."""

    module = HyperLoRALightningModule.load_from_checkpoint(hyper_ckpt, map_location=device)
    module = module.to(device)
    module.eval()
    module.model.eval()

    batch = _prepare_input(video_path, device, detector=detector)
    transcript, adapters = _predict_transcript(module, batch)

    video_frames, audio_frames, info = torchvision.io.read_video(
        video_path, pts_unit="sec", output_format="THWC"
    )
    captioned_frames = _draw_caption(video_frames, transcript)

    audio_array = audio_frames if audio_frames.numel() else None
    audio_fps = info.get("audio_fps", None) if audio_array is not None else None
    torchvision.io.write_video(
        output_path,
        captioned_frames,
        fps=info["video_fps"],
        audio_array=audio_array,
        audio_fps=audio_fps,
    )

    if save_adapter_path:
        torch.save(adapters, save_adapter_path)

    return transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render predicted transcript as captions")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument(
        "--hypernetwork-ckpt",
        required=True,
        help="Checkpoint from tutorials.hypernetwork_lora_train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output video path (defaults to <video>_captioned.mp4)",
    )
    parser.add_argument(
        "--save-adapter",
        type=str,
        default="",
        help="Optional path to save the predicted LoRA adapters as a .pt file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (defaults to cuda if available)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=["retinaface", "mediapipe"],
        help="Face detector used for preprocessing raw videos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_path = args.output_path
    if output_path is None:
        stem = os.path.splitext(args.video_path)[0]
        output_path = f"{stem}_captioned.mp4"

    transcript = render_captions(
        video_path=args.video_path,
        hyper_ckpt=args.hypernetwork_ckpt,
        output_path=output_path,
        device=device,
        detector=args.detector,
        save_adapter_path=args.save_adapter,
    )

    print(f"Transcript: {transcript}")
    print(f"Captioned video saved to {output_path}")
    if args.save_adapter:
        print(f"Adapters saved to {args.save_adapter}")


if __name__ == "__main__":
    main()
