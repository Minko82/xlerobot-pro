"""Visualize color detection results overlaid on the captured image.

Usage:
    python visualize_color_detect.py --color red
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from color_detect import detect_color, detection_to_xyz, COLOR_RANGES
from frame_transform.frame_transform import camera_xyz_to_base_xyz, camera_xyz_to_base2_xyz

CAPTURES_DIR = Path(__file__).resolve().parent / "outputs" / "realsense_capture"
DEG2RAD = np.pi / 180.0


def visualize(color: str = "red", head_pan_deg: float = 0.0,
              head_tilt_deg: float = 0.0, captures_dir: Path = CAPTURES_DIR,
              out_name: str = "color_detect_vis.png", show_window: bool = False,
              window_ms: int = 1200, exclude_bottom_fraction: float = 0.0):
    bgr = cv2.imread(str(captures_dir / "color.png"))
    if bgr is None:
        raise FileNotFoundError(f"color.png not found in {captures_dir}")
    depth_m = np.load(captures_dir / "depth_meters.npy")
    with open(captures_dir / "intrinsic_data.json") as f:
        intrinsics = json.load(f)

    joint_values = {
        "head_pan_joint": head_pan_deg * DEG2RAD,
        "head_tilt_joint": head_tilt_deg * DEG2RAD,
    }

    detections = detect_color(bgr, color, exclude_bottom_fraction=exclude_bottom_fraction)
    vis = bgr.copy()
    if exclude_bottom_fraction > 0.0:
        h = vis.shape[0]
        cutoff = int(h * (1.0 - exclude_bottom_fraction))
        cutoff = max(0, min(h, cutoff))
        cv2.line(vis, (0, cutoff), (vis.shape[1] - 1, cutoff), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"Bottom {exclude_bottom_fraction*100:.0f}% ignored",
            (10, max(20, cutoff - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

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

            # 3D coordinates in both frames
            try:
                xyz = detection_to_xyz(det, depth_m, intrinsics)
                cam_label = f"cam:({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
                bx, by, bz = camera_xyz_to_base_xyz(
                    xyz[0], xyz[1], xyz[2], joint_values)
                b2x, b2y, b2z = camera_xyz_to_base2_xyz(
                    xyz[0], xyz[1], xyz[2], joint_values)
                arm_label = f"base:({bx:.3f},{by:.3f},{bz:.3f})"
                arm2_label = f"base2:({b2x:.3f},{b2y:.3f},{b2z:.3f})"
            except RuntimeError:
                cam_label = "cam: no depth"
                arm_label = ""
                arm2_label = ""

            # Put labels above the bounding box
            cv2.putText(vis, f"#{i+1} {cam_label}", (x, y - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            if arm_label:
                cv2.putText(vis, arm_label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
            if arm2_label:
                cv2.putText(vis, arm2_label, (x, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 2)

        # Header
        cv2.putText(vis, f"Detected {len(detections)} {color} object(s)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = captures_dir / out_name
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")
    if show_window:
        try:
            # OpenCV path: timed popup only. Persistent mode is handled by eog fallback.
            if window_ms > 0:
                cv2.imshow("Color Detection", vis)
                cv2.waitKey(window_ms)
                cv2.destroyWindow("Color Detection")
            else:
                raise cv2.error("highgui", "imshow", "Persistent mode requested", -1)
        except cv2.error as e:
            print(f"OpenCV popup unavailable, trying eog fallback: {e}")
            eog = shutil.which("eog")
            if eog is None:
                print("eog not found on PATH, skipping popup display.")
                return
            try:
                proc = subprocess.Popen(
                    [eog, str(out_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if window_ms > 0:
                    time.sleep(window_ms / 1000.0)
                    if proc.poll() is None:
                        proc.terminate()
                else:
                    print("Opened visualization in eog (left running).")
            except Exception as ex:
                print(f"eog fallback failed, skipping popup: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize color detection")
    parser.add_argument("--color", default="red", choices=list(COLOR_RANGES.keys()))
    parser.add_argument("--head-pan", type=float, default=0.0,
                        help="Head pan angle in degrees (motor convention)")
    parser.add_argument("--head-tilt", type=float, default=0.0,
                        help="Head tilt angle in degrees (motor convention)")
    parser.add_argument("--out-name", default="color_detect_vis.png",
                        help="Output filename written under captures directory")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization window briefly after saving")
    parser.add_argument("--window-ms", type=int, default=1200,
                        help="How long to display window when --show is set")
    parser.add_argument("--exclude-bottom-fraction", type=float, default=0.0,
                        help="Ignore detections in bottom image fraction (0.0-1.0)")
    args = parser.parse_args()
    visualize(
        args.color,
        head_pan_deg=args.head_pan,
        head_tilt_deg=args.head_tilt,
        out_name=args.out_name,
        show_window=args.show,
        window_ms=args.window_ms,
        exclude_bottom_fraction=args.exclude_bottom_fraction,
    )
