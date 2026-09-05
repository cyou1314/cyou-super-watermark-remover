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
        description="Run a fixed rectangle or PNG-mask OpenCV inpainting baseline on a video."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--method", required=True, choices=("telea", "ns"))
    parser.add_argument("--radius", type=float, default=3.0)
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


def validate_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    resolved_input = input_path.resolve(strict=True)
    resolved_output = output_path.resolve(strict=False)
    if resolved_input == resolved_output:
        raise ValueError("Output must be different from input.")
    if resolved_output.exists():
        raise FileExistsError(f"Output already exists: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_input, resolved_output


def main() -> int:
    args = parse_args()
    if args.radius <= 0:
        raise ValueError("Inpainting radius must be greater than zero.")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH.")

    input_path, output_path = validate_paths(args.input, args.output)
    stream = probe_video(ffprobe, input_path)
    width = int(stream["width"])
    height = int(stream["height"])
    rectangle_values = (args.x, args.y, args.width, args.height)
    if args.mask is not None:
        if any(value is not None for value in rectangle_values):
            raise ValueError("Use either --mask or rectangle coordinates, not both.")
        mask_path = args.mask.resolve(strict=True)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"OpenCV could not read mask: {mask_path}")
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match video {width}x{height}."
            )
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        nonzero = cv2.findNonZero(mask)
        if nonzero is None:
            raise ValueError("Mask contains no selected pixels.")
        mask_x, mask_y, mask_width, mask_height = cv2.boundingRect(nonzero)
        mask_source = "image"
    else:
        if any(value is None for value in rectangle_values):
            raise ValueError("Rectangle mode requires --x, --y, --width and --height.")
        mask_x = int(args.x)
        mask_y = int(args.y)
        mask_width = int(args.width)
        mask_height = int(args.height)
        if mask_x < 0 or mask_y < 0 or mask_width < 1 or mask_height < 1:
            raise ValueError("Mask coordinates and dimensions must be positive.")
        if mask_x + mask_width > width or mask_y + mask_height > height:
            raise ValueError(f"Mask exceeds video bounds {width}x{height}.")
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[mask_y : mask_y + mask_height, mask_x : mask_x + mask_width] = 255
        mask_source = "rectangle"

    frame_rate = stream["avg_frame_rate"]
    numerator, denominator = (int(part) for part in frame_rate.split("/"))
    fps = numerator / denominator
    if fps <= 0:
        raise ValueError(f"Invalid frame rate: {frame_rate}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input: {input_path}")

    margin = max(8, math.ceil(args.radius * 3))
    crop_x0 = max(0, mask_x - margin)
    crop_y0 = max(0, mask_y - margin)
    crop_x1 = min(width, mask_x + mask_width + margin)
    crop_y1 = min(height, mask_y + mask_height + margin)
    crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]

    algorithm = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS
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
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("Failed to open FFmpeg pipes.")

    own_process = psutil.Process(os.getpid())
    encoder_process = psutil.Process(process.pid)
    frame_count = 0
    peak_combined_rss = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            repaired_crop = cv2.inpaint(crop, crop_mask, args.radius, algorithm)
            crop[crop_mask > 0] = repaired_crop[crop_mask > 0]
            frame[crop_y0:crop_y1, crop_x0:crop_x1] = crop
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

    output_probe = run_json(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output_path)]
    )
    result = {
        "method": args.method,
        "radius": args.radius,
        "mask": {
            "source": mask_source,
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
        "output_probe": output_probe,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
