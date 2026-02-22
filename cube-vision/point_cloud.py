import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import open3d as o3d
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend; avoids segfault on headless Jetson
import matplotlib.pyplot as plt


class PointCloud:
    def __init__(self, captures_dir=None):
        if captures_dir is None:
            self.script_dir = Path(__file__).resolve().parent
            self.captures_dir = self.script_dir / "outputs" / "realsense_capture"
        else:
            self.captures_dir = Path(captures_dir)
        self.ply_path = self.captures_dir / "vision.ply"
        self.pcd = None
        self.inlier_cloud = None
        self.outlier_cloud = None
        self.plane_model = None

    def load_from_ply(self, ply_path):
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
        self.pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"Loaded point cloud with {len(self.pcd.points)} points from {ply_path}")

    def create_point_cloud_from_rgbd(self, scale_depth=1.0, truncate_depth=0.75, min_depth=0.35):
        # Load color image
        color_path = self.captures_dir / "color.png"
        if not color_path.exists():
            raise FileNotFoundError(f"Color image not found: {color_path}")
        color_img = o3d.io.read_image(str(color_path))
        print("Color image loaded")

        # Load depth data from numpy file
        depth_path = self.captures_dir / "depth_meters.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth data not found: {depth_path}")
        depth_data = np.load(depth_path)
        print("Depth data loaded")
        print(f"Depth range: min={depth_data.min():.4f}, max={depth_data.max():.4f}")
        print(f"Points <= 0.10: {np.sum(depth_data <= 0.10)}")
        print(f"Points between 0.10 and 0.15: {np.sum((depth_data > 0.10) & (depth_data <= 0.15))}")
        depth_data[depth_data <= min_depth] = 0.0
        # Load camera intrinsics
        intrinsic_path = self.captures_dir / "intrinsic_data.json"
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic data not found: {intrinsic_path}")
        with open(intrinsic_path) as f:
            intrinsics = json.load(f)
        print("Intrinsics loaded from JSON")

        # Create Open3D camera intrinsic
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsics["width"],
            height=intrinsics["height"],
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["ppx"],
            cy=intrinsics["ppy"],
        )
        print(o3d_intrinsic)

        # Create point cloud manually (avoids Open3D ARM/Jetson segfault in create_from_rgbd_image)
        color_np = np.asarray(color_img)  # H x W x 3, uint8 RGB
        h, w = depth_data.shape
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["ppx"]
        cy = intrinsics["ppy"]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_data.astype(np.float64)
        # Zero out points outside [min_depth, truncate_depth]
        z[(z <= 0) | (z > truncate_depth)] = 0.0
        mask = z > 0
        x = ((u - cx) * z / fx)[mask]
        y = ((v - cy) * z / fy)[mask]
        z_valid = z[mask]
        points = np.stack([x, y, z_valid], axis=1)
        colors = color_np[mask].astype(np.float64) / 255.0
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Point cloud created with {len(self.pcd.points)} points")

    def save_to_ply(self):
        o3d.io.write_point_cloud(str(self.ply_path), self.pcd)
        print(f"Saved point cloud to {self.ply_path}")

    def segment_plane(self, distance_threshold=0.02, ransac_n=3, num_iterations=10000):
        self.plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
        )
        [a, b, c, d] = self.plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        self.pcd = self.pcd.select_by_index(inliers, invert=True)

    def crop_above_plane(self, max_height=0.20):
        """Keep only points within max_height meters above the detected table plane."""
        a, b, c, d = self.plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        points = np.asarray(self.pcd.points)
        dist = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
        mask = np.abs(dist) < max_height
        self.pcd = self.pcd.select_by_index(np.where(mask)[0])
        print(f"After crop_above_plane: {len(self.pcd.points)} points")

    def crop_sides(self, x_range=(-0.15, 0.15), y_range=(-0.10, 0.10)):
        """Remove points outside the given X and Y bounds (meters, camera frame)."""
        points = np.asarray(self.pcd.points)
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        )
        self.pcd = self.pcd.select_by_index(np.where(mask)[0])
        print(f"After crop_sides: {len(self.pcd.points)} points")

    def segment_grippers(self):
        pass

    def dbscan_objects(self, min_points_per_object=2000, colorize=False):
        labels = np.array(self.pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        if labels.size == 0 or labels.max() < 0:
            return []
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        # Apply colors to visualize clusters (skip when not needed; avoids ARM/Open3D segfault)
        if colorize:
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Extract coordinates for each object
        points = np.asarray(self.pcd.points)
        objects = []
        for i in range(max_label + 1):
            cluster_mask = labels == i
            cluster_points = points[cluster_mask]
            centroid = cluster_points.mean(axis=0)
            if len(cluster_points) >= min_points_per_object:
                objects.append(
                    {
                        "label": i,
                        "centroid": centroid,
                        "points": cluster_points,
                        "num_points": len(cluster_points),
                    }
                )
                print(f"Object {i}: centroid={centroid}, {len(cluster_points)} points")

        return objects

    def visualize(self, window_name="Point Cloud Visualization"):
        o3d.visualization.draw_geometries([self.pcd], window_name=window_name, width=1024, height=768)


if __name__ == "__main__":
    # Create processor and load/create point cloud
    processor = PointCloud()

    processor.create_point_cloud_from_rgbd(scale_depth=1.0, truncate_depth=1.5)

    # Segment the table plane
    processor.segment_plane(distance_threshold=0.02)
    processor.crop_above_plane(max_height=0.20)
    processor.crop_sides(x_range=(-0.30, 0.30), y_range=(-0.20, 0.20))
    processor.dbscan_objects(colorize=True)
    processor.save_to_ply()

    # processor.dbscan_objects()
    # Visualize the segmentation
    processor.visualize()
