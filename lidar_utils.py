import h5py
import numpy as np
import pandas as pd

def load_h5_data(file_path, dataset_name="lidar_points"):
    """Loads HDF5 data into a Pandas DataFrame."""
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")
        points = f[dataset_name][:]
    
    return pd.DataFrame({name: points[name] for name in points.dtype.names})

def get_unique_poses(df):
    """Returns a DataFrame of unique poses with a 'pose_index'."""
    pose_fields = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
    if not all(f in df.columns for f in pose_fields):
        return None
    
    pose_counts = (
        df.groupby(pose_fields)
        .size()
        .reset_index(name="num_points")
        .reset_index(names="pose_index")
    )
    return pose_counts

def filter_by_pose(df, pose_row):
    """Filters the dataframe for a specific pose quadruplet."""
    return df[
        (df["ego_x"] == pose_row["ego_x"]) &
        (df["ego_y"] == pose_row["ego_y"]) &
        (df["ego_z"] == pose_row["ego_z"]) &
        (df["ego_yaw"] == pose_row["ego_yaw"])
    ].reset_index(drop=True)

def spherical_to_local_cartesian(df):
    """Converts spherical coordinates to local Cartesian (Lidar Frame)."""
    distance_m = df["distance_cm"].to_numpy() / 100.0
    azimuth_rad = np.radians(df["azimuth_raw"] / 100.0)
    elevation_rad = np.radians(df["elevation_raw"] / 100.0)

    x = distance_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = -distance_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance_m * np.sin(elevation_rad)

    return np.column_stack((x, y, z))