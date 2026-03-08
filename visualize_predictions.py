import argparse
import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize

import lidar_utils
from sanity_check_predictions import attach_pose_index


WINDOW_WIDTH, WINDOW_HEIGHT = 1440, 900
CLASS_COLORS = {
    "Antenna": [0.95, 0.30, 0.25],
    "Cable": [1.00, 0.78, 0.12],
    "Electric Pole": [0.20, 0.72, 0.35],
    "Wind Turbine": [0.16, 0.55, 0.95],
}
BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize point cloud with predicted bounding boxes")
    parser.add_argument("--file", required=True, help="Path to source HDF5 file")
    parser.add_argument("--pred", required=True, help="Path to prediction CSV")
    parser.add_argument("--pose-index", type=int, default=None,
                        help="Index of the unique pose to visualize (0-based)")
    parser.add_argument("--cmap", default="turbo", help="Colormap for reflectivity")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size")
    return parser.parse_args()


def rotation_matrix_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def bbox_corners(center, width, length, height, yaw):
    hx, hy, hz = width / 2.0, length / 2.0, height / 2.0
    local = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=np.float64)
    R = rotation_matrix_z(yaw)
    return local @ R.T + center


def make_box_lineset(det):
    center = np.array([
        det["bbox_center_x"],
        det["bbox_center_y"],
        det["bbox_center_z"],
    ], dtype=np.float64)
    corners = bbox_corners(
        center=center,
        width=float(det["bbox_width"]),
        length=float(det["bbox_length"]),
        height=float(det["bbox_height"]),
        yaw=float(det["bbox_yaw"]),
    )

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    color = CLASS_COLORS.get(det["class_label"], [1.0, 1.0, 1.0])
    lineset.colors = o3d.utility.Vector3dVector([color] * len(BOX_EDGES))
    return lineset


def load_pose_table(file_path):
    df = lidar_utils.load_h5_data(file_path)
    if len(df) == 0:
        raise ValueError("Dataset contains 0 lidar points.")
    poses = lidar_utils.get_unique_poses(df)
    if poses is None:
        raise ValueError("Pose fields not found in dataset.")
    return df, poses


def prediction_table(pred_path, poses):
    pred_df = pd.read_csv(pred_path)
    if len(pred_df) == 0:
        pred_df = pred_df.copy()
        pred_df["pose_index"] = pd.Series(dtype=int)
        return pred_df
    return attach_pose_index(pred_df, poses)


def print_pose_table(poses, pred_df):
    pose_counts = (
        pred_df.groupby("pose_index")
        .size()
        .rename("num_detections")
        .reset_index()
        if len(pred_df) > 0 else pd.DataFrame(columns=["pose_index", "num_detections"])
    )
    merged = poses.merge(pose_counts, on="pose_index", how="left").fillna({"num_detections": 0})
    merged["num_detections"] = merged["num_detections"].astype(int)
    print(merged[[
        "pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points", "num_detections"
    ]].to_string(index=False, float_format="%.2f"))
    print("\nUse '--pose-index N' to visualize a specific pose.")


def build_point_cloud(frame_df, cmap_name):
    xyz = lidar_utils.spherical_to_local_cartesian(frame_df)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if {"r", "g", "b"}.issubset(frame_df.columns) and not (
        (frame_df["r"] == 128).all() and (frame_df["g"] == 128).all() and (frame_df["b"] == 128).all()
    ):
        rgb = np.column_stack((
            frame_df["r"].to_numpy() / 255.0,
            frame_df["g"].to_numpy() / 255.0,
            frame_df["b"].to_numpy() / 255.0,
        ))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    elif "reflectivity" in frame_df.columns:
        intensities = frame_df["reflectivity"].to_numpy()
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())
        colors = colormaps.get_cmap(cmap_name)(norm(intensities))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.75, 0.75, 0.75])

    return pcd, xyz


def print_detection_summary(frame_preds):
    if len(frame_preds) == 0:
        print("No detections for this pose.")
        return

    counts = frame_preds["class_label"].value_counts()
    print(f"Detections: {len(frame_preds)}")
    for label in ("Antenna", "Cable", "Electric Pole", "Wind Turbine"):
        value = int(counts.get(label, 0))
        if value > 0:
            print(f"  {label}: {value}")

    cols = [
        "class_label", "bbox_center_x", "bbox_center_y", "bbox_center_z",
        "bbox_width", "bbox_length", "bbox_height", "bbox_yaw",
    ]
    preview = frame_preds[cols].copy()
    print("\nPreview:")
    print(preview.to_string(index=False, float_format="%.2f"))


def main():
    args = parse_args()

    try:
        df, poses = load_pose_table(args.file)
        pred_df = prediction_table(args.pred, poses)
    except Exception as e:
        print(f"Error: {e}")
        return

    if args.pose_index is None:
        print_pose_table(poses, pred_df)
        return

    if args.pose_index < 0 or args.pose_index >= len(poses):
        print(f"Invalid pose index {args.pose_index}. File has {len(poses)} unique poses.")
        return

    selected_pose = poses.iloc[args.pose_index]
    frame_df = lidar_utils.filter_by_pose(df, selected_pose)
    frame_preds = pred_df[pred_df["pose_index"] == args.pose_index].reset_index(drop=True)

    print(poses.loc[poses["pose_index"] == args.pose_index, [
        "pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"
    ]].to_string(index=False, float_format="%.2f"))
    print(f"\nSelected pose #{args.pose_index} -> {len(frame_df)} lidar points")
    print_detection_summary(frame_preds)

    pcd, xyz = build_point_cloud(frame_df, args.cmap)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Predicted Boxes", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    vis.add_geometry(pcd)

    added = 0
    for _, det in frame_preds.iterrows():
        vis.add_geometry(make_box_lineset(det))
        added += 1

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord)

    pts = np.asarray(pcd.points)
    print(f"\nBounds X:[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] "
          f"Y:[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] "
          f"Z:[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")
    print(f"Rendered boxes: {added}")

    ctrl = vis.get_view_control()
    cam_pos = np.array([0.0, 0.0, 0.0])
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    lookat = cam_pos + 12.0 * forward
    ctrl.set_lookat(lookat)
    ctrl.set_front(-forward)
    ctrl.set_up(up)
    ctrl.set_zoom(0.08)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(args.point_size)
    render_opt.line_width = 2.0

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
