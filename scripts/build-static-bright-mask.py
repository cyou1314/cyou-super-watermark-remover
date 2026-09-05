from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed bright low-saturation watermark mask from a video."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--x", required=True, type=int)
    parser.add_argument("--y", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--luma-percentile", type=float, default=10.0)
    parser.add_argument("--minimum-luma", type=int, default=150)
    parser.add_argument("--saturation-percentile", type=float, default=90.0)
    parser.add_argument("--maximum-saturation", type=int, default=70)
    parser.add_argument("--dilate", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve(strict=True)
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if args.x < 0 or args.y < 0 or args.width < 1 or args.height < 1:
        raise ValueError("Mask coordinates and dimensions must be positive.")
    if not 0 <= args.luma_percentile <= 100:
        raise ValueError("Luma percentile must be between 0 and 100.")
    if not 0 <= args.saturation_percentile <= 100:
        raise ValueError("Saturation percentile must be between 0 and 100.")
    if not 0 <= args.minimum_luma <= 255 or not 0 <= args.maximum_saturation <= 255:
        raise ValueError("Luma and saturation thresholds must be between 0 and 255.")
    if args.dilate < 0:
        raise ValueError("Dilation iterations must be zero or greater.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input: {input_path}")

    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.x + args.width > video_width or args.y + args.height > video_height:
        raise ValueError(f"Search rectangle exceeds video bounds {video_width}x{video_height}.")

    luma_frames: list[np.ndarray] = []
    saturation_frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            roi = frame[args.y : args.y + args.height, args.x : args.x + args.width]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            luma_frames.append(gray)
            saturation_frames.append(hsv[:, :, 1])
    finally:
        capture.release()

    if not luma_frames:
        raise RuntimeError("No video frames were decoded.")

    luma_floor = np.percentile(
        np.stack(luma_frames), args.luma_percentile, axis=0
    )
    saturation_ceiling = np.percentile(
        np.stack(saturation_frames), args.saturation_percentile, axis=0
    )
    roi_mask = np.where(
        (luma_floor >= args.minimum_luma)
        & (saturation_ceiling <= args.maximum_saturation),
        255,
        0,
    ).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel)
    if args.dilate:
        roi_mask = cv2.dilate(roi_mask, close_kernel, iterations=args.dilate)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask)
    cleaned = np.zeros_like(roi_mask)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == component] = 255
    roi_mask = cleaned

    mask_pixels = int(np.count_nonzero(roi_mask))
    if mask_pixels == 0:
        raise RuntimeError("No watermark pixels matched the configured thresholds.")

    full_mask = np.zeros((video_height, video_width), dtype=np.uint8)
    full_mask[args.y : args.y + args.height, args.x : args.x + args.width] = roi_mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), full_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"Failed to write mask: {output_path}")

    result = {
        "frames_analyzed": len(luma_frames),
        "video_size": {"width": video_width, "height": video_height},
        "search_rectangle": {
            "x": args.x,
            "y": args.y,
            "width": args.width,
            "height": args.height,
            "pixels": args.width * args.height,
        },
        "mask_pixels": mask_pixels,
        "mask_to_rectangle_ratio": round(mask_pixels / (args.width * args.height), 6),
        "parameters": {
            "luma_percentile": args.luma_percentile,
            "minimum_luma": args.minimum_luma,
            "saturation_percentile": args.saturation_percentile,
            "maximum_saturation": args.maximum_saturation,
            "dilate": args.dilate,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
