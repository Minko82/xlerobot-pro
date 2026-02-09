import open3d as o3d
import os

# Get the path to the point cloud file
current_dir = os.path.dirname(os.path.abspath(__file__))
ply_path = os.path.join(current_dir, "outputs", "realsense_capture", "vision.ply")

# Load the point cloud
print(f"Loading point cloud from: {ply_path}")
pcd = o3d.io.read_point_cloud(ply_path)

# Print some info about the point cloud
print(f"Point cloud has {len(pcd.points)} points")
print(f"Point cloud has colors: {pcd.has_colors()}")

# Visualize the point cloud
print("Displaying point cloud...")
o3d.visualization.draw_geometries(
    [pcd], window_name="Point Cloud Visualization", width=1024, height=768, point_show_normal=False
)
