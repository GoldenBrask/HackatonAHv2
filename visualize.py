import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
import lidar_utils

window_width, window_height = 1280, 720

def main():
    parser = argparse.ArgumentParser(description="Visualize and export Lidar points from an HDF5 file")
    parser.add_argument("--file", required=True, help="Path to HDF5 lidar file")
    parser.add_argument("--pose-index", type=int, default=None,
                        help="Index of the unique pose to visualize (0-based)")
    parser.add_argument("--cmap", default="turbo", help="Colormap for intensity")

    args = parser.parse_args()

    # 1. Load Data
    try:
        df = lidar_utils.load_h5_data(args.file)
        print(f"Loaded {len(df)} points from {args.file}")
    except Exception as e:
        print(f"Error: {e}")
        return

    if len(df) == 0:
        print("Dataset contains 0 lidar points. Nothing to visualize.")
        return

    # 2. Handle Poses
    pose_counts = lidar_utils.get_unique_poses(df)

    if pose_counts is not None:
        if args.pose_index is None:
            # ---- No pose selected → show all poses ----
            print(pose_counts[["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]]
                  .to_string(index=False, float_format="%.2f"))
            print("\nUse '--pose-index N' to visualize a specific pose (0-based index).")
            return
        
        if args.pose_index < 0 or args.pose_index >= len(pose_counts):
            print(f"Invalid pose index {args.pose_index}. File has {len(pose_counts)} unique poses.")
            return

        # ---- Pose selected → filter ----
        print(pose_counts.loc[pose_counts["pose_index"] == args.pose_index,
                              ["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]]
              .to_string(index=False, float_format="%.2f"))

        selected_pose = pose_counts.iloc[args.pose_index]
        df = lidar_utils.filter_by_pose(df, selected_pose)
        print(f"\nSelected pose #{args.pose_index} → {len(df)} lidar points")

    else:
        print("Pose fields not found in dataset — cannot filter by pose index.")

    # 3. Convert Coordinates
    xyz = lidar_utils.spherical_to_local_cartesian(df)

    # 4. Visualization (Local Frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # --- Apply Coloring ---
    if {"r", "g", "b"}.issubset(df.columns):
        print("Using ground-truth RGB colors from dataset")
        rgb = np.column_stack((
            df["r"].to_numpy() / 255.0,
            df["g"].to_numpy() / 255.0,
            df["b"].to_numpy() / 255.0
        ))
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    elif "reflectivity" in df.columns:
        print("No RGB fields found → using reflectivity colormap")
        intensities = df["reflectivity"].to_numpy()
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())
        cmap = colormaps.get_cmap(args.cmap)
        colors = cmap(norm(intensities))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    else:
        print("No reflectivity or RGB → using red")
        pcd.paint_uniform_color([1, 0, 0])

    # --- Launch Viewer ---
    title = "All LiDAR points"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=window_width, height=window_height)
    vis.add_geometry(pcd)

    pts = np.asarray(pcd.points)
    print(f"Bounds X:[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] "
          f"Y:[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] "
          f"Z:[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")

    ctrl = vis.get_view_control()

    # Exact camera settings
    cam_pos = np.array([0.0, 0.0, 0.0])   # Lidar position in local frame
    forward = np.array([1.0, 0.0, 0.0])   # Lidar points forward along X-axis
    up = np.array([0.0, 0.0, 1.0])        # Z-axis is up
    
    lookat = cam_pos + 10.0 * forward

    ctrl.set_lookat(lookat)
    ctrl.set_front(-forward)
    ctrl.set_up(up)
    ctrl.set_zoom(0.1)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()