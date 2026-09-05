from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenCV inpainting followed by motion-compensated temporal "
            "stabilization inside a fixed PNG mask."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--method", choices=("telea", "ns"), default="telea")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--temporal-strength", type=float, default=0.65)
    parser.add_argument("--scene-threshold", type=float, default=24.0)
    parser.add_argument("--video-codec", choices=("h264", "h265"), default="h265")
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument(
        "--preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        default="medium",
    )
    return parser.parse_args()


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def probe_video(ffprobe: str, input_path: Path) -> dict:
    data = run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(input_path),
        ]
    )
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("Expected exactly one video stream.")
    return streams[0]


def validate_args(args: argparse.Namespace) -> None:
    if args.radius <= 0:
        raise ValueError("Inpainting radius must be greater than zero.")
    if args.temporal_window < 1:
        raise ValueError("Temporal window must be at least one frame.")
    if not 0 <= args.temporal_strength <= 1:
        raise ValueError("Temporal strength must be between zero and one.")
    if args.scene_threshold <= 0:
        raise ValueError("Scene threshold must be greater than zero.")
    if not 0 <= args.crf <= 51:
        raise ValueError("CRF must be between zero and 51.")


def validate_paths(
    input_path: Path, output_path: Path, mask_path: Path
) -> tuple[Path, Path, Path]:
    resolved_input = input_path.resolve(strict=True)
    resolved_output = output_path.resolve(strict=False)
    resolved_mask = mask_path.resolve(strict=True)
    if resolved_input == resolved_output:
        raise ValueError("Output must be different from input.")
    if resolved_output.exists():
        raise FileExistsError(f"Output already exists: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_input, resolved_output, resolved_mask


def repair_crops(
    input_path: Path,
    crop_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    radius: float,
    algorithm: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    crop_x0, crop_y0, crop_x1, crop_y1 = bounds
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input: {input_path}")

    repaired_crops: list[np.ndarray] = []
    motion_grays: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            repaired = cv2.inpaint(crop, crop_mask, radius, algorithm)
            crop[crop_mask > 0] = repaired[crop_mask > 0]
            repaired_crops.append(crop)
            motion_grays.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    if not repaired_crops:
        raise RuntimeError("No video frames were decoded.")
    return repaired_crops, motion_grays


def stabilize_crops(
    repaired_crops: list[np.ndarray],
    motion_grays: list[np.ndarray],
    crop_mask: np.ndarray,
    temporal_window: int,
    temporal_strength: float,
    scene_threshold: float,
) -> tuple[list[np.ndarray], int, int]:
    crop_height, crop_width = crop_mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(crop_width, dtype=np.float32),
        np.arange(crop_height, dtype=np.float32),
    )
    outside_mask = crop_mask == 0
    stabilized: list[np.ndarray] = []
    accepted_neighbors = 0
    rejected_neighbors = 0

    for frame_index, current in enumerate(repaired_crops):
        candidates = [current.astype(np.float32), current.astype(np.float32)]
        current_gray = motion_grays[frame_index]
        start = max(0, frame_index - temporal_window)
        end = min(len(repaired_crops), frame_index + temporal_window + 1)

        for neighbor_index in range(start, end):
            if neighbor_index == frame_index:
                continue
            neighbor_gray = motion_grays[neighbor_index]
            flow = cv2.calcOpticalFlowFarneback(
                current_gray,
                neighbor_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            map_x = grid_x + flow[:, :, 0]
            map_y = grid_y + flow[:, :, 1]
            warped = cv2.remap(
                repaired_crops[neighbor_index],
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            warped_gray = cv2.remap(
                neighbor_gray,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            alignment_error = float(
                np.mean(
                    cv2.absdiff(current_gray, warped_gray)[outside_mask],
                    dtype=np.float64,
                )
            )
            if alignment_error > scene_threshold:
                rejected_neighbors += 1
                continue
            candidates.append(warped.astype(np.float32))
            accepted_neighbors += 1

        temporal_median = np.median(np.stack(candidates), axis=0)
        blended = cv2.addWeighted(
            current.astype(np.float32),
            1.0 - temporal_strength,
            temporal_median.astype(np.float32),
            temporal_strength,
            0,
        )
        result = current.copy()
        selected = crop_mask > 0
        result[selected] = np.clip(blended[selected], 0, 255).astype(np.uint8)
        stabilized.append(result)

    return stabilized, accepted_neighbors, rejected_neighbors


def build_encoder_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    frame_rate: str,
    video_codec: str,
    crf: int,
    preset: str,
) -> list[str]:
    encoder = "libx264" if video_codec == "h264" else "libx265"
    command = [
        ffmpeg,
        "-hide_banner",
        "-n",
        "-benchmark",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        frame_rate,
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        encoder,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
    ]
    if video_codec == "h265":
        command.extend(("-tag:v", "hvc1"))
    command.append(str(output_path))
    return command


def main() -> int:
    args = parse_args()
    validate_args(args)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")

    input_path, output_path, mask_path = validate_paths(
        args.input, args.output, args.mask
    )
    stream = probe_video(ffprobe, input_path)
    width = int(stream["width"])
    height = int(stream["height"])
    frame_rate = stream["avg_frame_rate"]
    numerator, denominator = (int(part) for part in frame_rate.split("/"))
    fps = numerator / denominator
    if fps <= 0:
        raise ValueError(f"Invalid frame rate: {frame_rate}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"OpenCV could not read mask: {mask_path}")
    if mask.shape != (height, width):
        raise ValueError(
            f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match "
            f"video {width}x{height}."
        )
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        raise ValueError("Mask contains no selected pixels.")
    mask_x, mask_y, mask_width, mask_height = cv2.boundingRect(nonzero)

    margin = max(24, math.ceil(args.radius * 6))
    crop_x0 = max(0, mask_x - margin)
    crop_y0 = max(0, mask_y - margin)
    crop_x1 = min(width, mask_x + mask_width + margin)
    crop_y1 = min(height, mask_y + mask_height + margin)
    bounds = (crop_x0, crop_y0, crop_x1, crop_y1)
    crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    algorithm = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS

    own_process = psutil.Process(os.getpid())
    peak_combined_rss = own_process.memory_info().rss
    started = time.perf_counter()
    repaired_crops, motion_grays = repair_crops(
        input_path, crop_mask, bounds, args.radius, algorithm
    )
    stabilized_crops, accepted_neighbors, rejected_neighbors = stabilize_crops(
        repaired_crops,
        motion_grays,
        crop_mask,
        args.temporal_window,
        args.temporal_strength,
        args.scene_threshold,
    )
    peak_combined_rss = max(peak_combined_rss, own_process.memory_info().rss)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not reopen input: {input_path}")
    command = build_encoder_command(
        ffmpeg,
        input_path,
        output_path,
        width,
        height,
        frame_rate,
        args.video_codec,
        args.crf,
        args.preset,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("Failed to open FFmpeg pipes.")
    encoder_process = psutil.Process(process.pid)
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_count >= len(stabilized_crops):
                raise RuntimeError("Second decode produced more frames than the first pass.")
            crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            selected = crop_mask > 0
            crop[selected] = stabilized_crops[frame_count][selected]
            process.stdin.write(frame.tobytes())
            frame_count += 1
            try:
                combined_rss = (
                    own_process.memory_info().rss + encoder_process.memory_info().rss
                )
                peak_combined_rss = max(peak_combined_rss, combined_rss)
            except psutil.Error:
                pass
    finally:
        capture.release()
        process.stdin.close()
        process.stdin = None

    _, stderr_bytes = process.communicate()
    elapsed = time.perf_counter() - started
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    sys.stderr.write(stderr_text)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {process.returncode}.")
    if frame_count != len(stabilized_crops):
        raise RuntimeError(
            f"Frame count changed between passes: {len(stabilized_crops)} vs {frame_count}."
        )

    output_probe = run_json(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output_path)]
    )
    result = {
        "method": args.method,
        "radius": args.radius,
        "temporal": {
            "window": args.temporal_window,
            "strength": args.temporal_strength,
            "scene_threshold": args.scene_threshold,
            "accepted_neighbors": accepted_neighbors,
            "rejected_neighbors": rejected_neighbors,
        },
        "mask": {
            "x": mask_x,
            "y": mask_y,
            "width": mask_width,
            "height": mask_height,
            "pixels": int(np.count_nonzero(mask)),
        },
        "frames_processed": frame_count,
        "elapsed_seconds": round(elapsed, 6),
        "processing_fps": round(frame_count / elapsed, 3),
        "realtime_factor": round((frame_count / fps) / elapsed, 3),
        "peak_combined_rss_bytes": peak_combined_rss,
        "output_bytes": output_path.stat().st_size,
        "opencv_version": cv2.__version__,
        "video_encoder": {
            "codec": args.video_codec,
            "crf": args.crf,
            "preset": args.preset,
            "pixel_format": "yuv420p",
        },
        "output_probe": output_probe,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
