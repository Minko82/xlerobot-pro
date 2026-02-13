import os
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
import json

out_dir = Path("outputs/realsense_capture")
out_dir.mkdir(parents=True, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()

# Enable BOTH streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Depth scale (meters per unit)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth scale:", depth_scale, "meters/unit")

# Align depth to color
align = rs.align(rs.stream.color)

# Warmup a bit
for _ in range(30):
    frames = pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
frames = align.process(frames)

depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

if not depth_frame or not color_frame:
    pipeline.stop()
    raise RuntimeError("Failed to get depth or color frame")

depth_image = np.asanyarray(depth_frame.get_data())   # uint16
color_image = np.asanyarray(color_frame.get_data())   # uint8 BGR

# Save color
cv2.imwrite(str(out_dir / "color.png"), color_image)

# Save raw depth
cv2.imwrite(str(out_dir / "depth_16u.png"), depth_image)

# Also save meters as float32
depth_m = depth_image.astype(np.float32) * depth_scale
np.save(out_dir / "depth_meters.npy", depth_m)

# visualization of depth
depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = depth_vis.astype(np.uint8)
cv2.imwrite(str(out_dir / "depth_vis.png"), depth_vis)

# 3. Extract Intrinsics
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# 4. create a dictionary of the values
intrinsic_data = {
    "width": intrinsics.width,
    "height": intrinsics.height,
    "fx": intrinsics.fx,
    "fy": intrinsics.fy,
    "ppx": intrinsics.ppx,
    "ppy": intrinsics.ppy,
    "model": str(intrinsics.model),
}

intrinsics_path = os.path.join(out_dir, "intrinsic_data.json")

# 5. Save to JSON
with open(intrinsics_path, "w") as f:
    json.dump(intrinsic_data, f, indent=4)

pipeline.stop()
print("Saved to:", out_dir)

