"""Visualize color detection results overlaid on the captured image.

Usage:
    python visualize_color_detect.py --color red
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from color_detect import detect_color, detection_to_xyz, COLOR_RANGES

CAPTURES_DIR = Path(__file__).resolve().parent / "outputs" / "realsense_capture"


def visualize(color: str = "red", captures_dir: Path = CAPTURES_DIR):
    bgr = cv2.imread(str(captures_dir / "color.png"))
    if bgr is None:
        raise FileNotFoundError(f"color.png not found in {captures_dir}")
    depth_m = np.load(captures_dir / "depth_meters.npy")
    with open(captures_dir / "intrinsic_data.json") as f:
        intrinsics = json.load(f)

    detections = detect_color(bgr, color)
    vis = bgr.copy()

    if not detections:
        cv2.putText(vis, f"No {color} objects found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        for i, det in enumerate(detections):
            # Bounding box
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Contour outline
            cv2.drawContours(vis, [det.contour], -1, (255, 255, 0), 1)

            # Centroid dot
            cx, cy = det.centroid_px
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)

            # 3D coordinates
            try:
                xyz = detection_to_xyz(det, depth_m, intrinsics)
                label = f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})m"
            except RuntimeError:
                label = "no depth"

            # Put label above the bounding box
            cv2.putText(vis, f"#{i+1} {label}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Header
        cv2.putText(vis, f"Detected {len(detections)} {color} object(s)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = captures_dir / "color_detect_vis.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize color detection")
    parser.add_argument("--color", default="red", choices=list(COLOR_RANGES.keys()))
    args = parser.parse_args()
    visualize(args.color)
