"""Generate an overlaid transcription video using the inference notebook pipeline.

This script loads a checkpoint, runs inference on an input video (or audio
file, depending on the selected modality), and writes a copy of the video with
the predicted transcript rendered as subtitles.
"""

import argparse
import os
import textwrap
from typing import List, Optional, Tuple

import cv2
import torch
import torchaudio
import torchvision

from datamodule.transforms import AudioTransform, VideoTransform
from lightning import ModelModule


class InferencePipeline(torch.nn.Module):
    """A thin wrapper around ``ModelModule`` used in the tutorial notebook."""

    def __init__(self, args, ckpt_path: str, detector: str = "mediapipe", device: Optional[str] = None):
        super().__init__()
        self.modality = args.modality
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.modality == "audio":
            self.audio_transform = AudioTransform(subset="test")
        elif self.modality == "video":
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device=str(self.device))
                self.video_process = VideoProcess(convert_gray=False)
            else:
                raise ValueError(f"Unknown detector '{detector}'")
            self.video_transform = VideoTransform(subset="test")

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(args).to(self.device)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def forward(self, data_filename: str) -> str:
        transcript, _, _ = self.transcribe(data_filename, return_frames=False)
        return transcript

    def transcribe(self, data_filename: str, return_frames: bool = False) -> Tuple[str, Optional[torch.Tensor], Optional[float]]:
        data_filename = os.path.abspath(data_filename)
        if not os.path.isfile(data_filename):
            raise FileNotFoundError(f"data_filename: {data_filename} does not exist.")

        original_frames = None
        fps = None

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio).to(self.device)
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        elif self.modality == "video":
            video, _, info = torchvision.io.read_video(data_filename, pts_unit="sec")
            fps = info.get("video_fps")
            if return_frames:
                original_frames = video.clone()

            video_np = video.numpy()
            landmarks = self.landmarks_detector(video_np)
            video_processed = self.video_process(video_np, landmarks)
            video_tensor = torch.tensor(video_processed)
            video_tensor = video_tensor.permute((0, 3, 1, 2))
            video_tensor = self.video_transform(video_tensor).to(self.device)

            with torch.no_grad():
                transcript = self.modelmodule(video_tensor)
        else:
            raise ValueError(f"Unsupported modality '{self.modality}'")

        return transcript, original_frames, fps

    @staticmethod
    def load_audio(data_filename: str):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    @staticmethod
    def audio_process(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = 16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


def overlay_transcript_on_frames(
    frames: torch.Tensor,
    transcript: str,
    font_scale: float = 0.8,
    thickness: int = 2,
    wrap_width: int = 48,
    margin: int = 20,
    line_spacing: int = 6,
) -> torch.Tensor:
    """Render the transcript text on every frame of the video."""

    if transcript.strip() == "":
        return frames

    wrapped_lines = textwrap.wrap(transcript, width=wrap_width) or [transcript]
    overlaid_frames: List[torch.Tensor] = []

    for frame in frames:
        frame_np = frame.cpu().numpy()
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        line_heights = []
        for line in wrapped_lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            line_heights.append(text_height)

        total_height = sum(line_heights) + line_spacing * (len(wrapped_lines) - 1)
        y_start = frame_bgr.shape[0] - margin - total_height

        y = y_start
        for line, text_height in zip(wrapped_lines, line_heights):
            cv2.putText(
                frame_bgr,
                line,
                (margin, y + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            y += text_height + line_spacing

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        overlaid_frames.append(torch.from_numpy(frame_rgb))

    return torch.stack(overlaid_frames)


def write_video(frames: torch.Tensor, fps: float, output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torchvision.io.write_video(output_path, frames, fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and overlay transcriptions on a video.")
    parser.add_argument("input", help="Path to the input video (or audio if modality=audio).")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint (.pth file).")
    parser.add_argument("--output", default="output_with_transcript.mp4", help="Filename for the subtitled video.")
    parser.add_argument(
        "--modality",
        default="video",
        choices=["audio", "video"],
        help="Inference modality. 'video' overlays on the input video; 'audio' produces an audio-only transcript.",
    )
    parser.add_argument(
        "--detector",
        default="mediapipe",
        choices=["mediapipe", "retinaface"],
        help="Face landmark detector used for visual inference.",
    )
    parser.add_argument("--device", default=None, help="Torch device to run inference on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--font-scale", type=float, default=0.6, help="OpenCV font scale for the overlaid transcript text.")
    parser.add_argument("--font-thickness", type=int, default=1, help="OpenCV font thickness for the overlaid transcript text.")
    parser.add_argument("--wrap-width", type=int, default=12, help="Number of characters per line before wrapping the transcript.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ``ModelModule`` expects an argparse-style namespace so we mirror that here
    # instead of using ``types.SimpleNamespace`` (which ``save_hyperparameters``
    # does not support).
    inference_args = argparse.Namespace(modality=args.modality)
    pipeline = InferencePipeline(inference_args, args.checkpoint, detector=args.detector, device=args.device)

    transcript, frames, fps = pipeline.transcribe(args.input, return_frames=True)
    print(f"Predicted transcript: {transcript}")

    if args.modality == "video":
        if frames is None or fps is None:
            raise RuntimeError("Video frames were not returned for rendering.")
        subtitled_frames = overlay_transcript_on_frames(
            frames,
            transcript,
            font_scale=args.font_scale,
            thickness=args.font_thickness,
            wrap_width=args.wrap_width,
        )
        write_video(subtitled_frames, fps, args.output)
        print(f"Saved subtitled video to {os.path.abspath(args.output)}")
    else:
        print("Audio-only modality selected. No video written; transcript printed above.")


if __name__ == "__main__":
    main()
