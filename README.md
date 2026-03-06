# Airbus AI Hackathon 2026: Lidar Obstacle Detection Toolkit

## Repository Note
This public repository contains the source code, documentation, and README assets only.
The Airbus training/evaluation datasets, generated checkpoints, processed tensors, logs, and predictions are intentionally excluded.

Welcome to the 2026 Airbus AI Hackathon! We trust you're more ready than ever to start this challenge. To assist you, this toolkit provides the scripts necessary to load, manipulate, and visualize the simulated Lidar data.

## 🚁 The Challenge
**Context:** Collision with obstacles, particularly power lines, is a major safety priority for low-altitude helicopter flights.

**Objective:** Develop an algorithm to **detect and classify 3D obstacles** from Lidar point clouds.

**Target Classes**
You must detect the following 4 categories:

| Class ID | Label | Description |
| :---: | :--- | :--- |
| **0** | **Antenna** | Communication masts and antennas. |
| **1** | **Cable** | Power lines (often thin and difficult to detect). |
| **2** | **Electric pole** | Pylons and utility poles supporting cables. |
| **3** | **Wind turbine** | Large wind energy turbines. |

> **⚠️ Important:** Ground truth 3D bounding boxes are **not included** in the supplied files. You must reconstruct them yourself using the point-wise color labels provided in the data.

---

## 💾 Data Specifications

### 1. Data generation process

The dataset is generated via simulation software:
1.  **Scene Generation:** Objects of interest (i.e. obstacles) are positioned on a map containing terrain (ground, trees, rocks, grass). The training dataset comprises **10 different scenes**.
2.  **Lidar Simulation:** The sensor is integrated into a vehicle called "Ego," which is "teleported" to various positions.
    * The training dataset contains **100 unique frames** (unique Ego localizations).
    * Frames may contain multiple obstacles or none at all.
3.  **Point Cloud:** Each frame simulation runs for ~1 second, generating up to **575,000 points**. Invalid beams (no return) are identified with distance_cm=0, so valid point counts vary per frame.

<div align="center">
  <img src="images/cam_image.bmp" width="50%">
  <br>
  <sup>Camera example of one scene frame generated.</sup>
</div>

<table style="width:100%; text-align:center;">
  <tr>
    <td style="width:39%;">
      <img src="images/lidar_ex1.png" style="width:100%;">
      <br>
      <sup>Example of Lidar visualization with electric poles and cables.</sup>
    </td>
    <td style="width:25%;">
      <img src="images/lidar_ex2.png" style="width:100%;">
      <br>
      <sup>Example of Lidar visualization with wind turbine and antenna.</sup>
    </td>
  </tr>
</table>

### 2. File Structure

The training dataset consists of **10 HDF5 files**. 
Each file contains concatenated frames.

**Field Definitions:**

| Field | Description | Unit / Format |
| :--- | :--- | :--- |
| **`distance_cm`** | Distance from Lidar origin to hit point. | Centimeters |
| **`azimuth_raw`** | Horizontal angle (FOV center to hit point). | Hundredth of degree |
| **`elevation_raw`** | Vertical angle (FOV center to hit point). | Hundredth of degree |
| **`reflectivity`** | Intensity of returned laser beam. | 8-bit integer (0-255) |
| **`r, g, b`** | Ground Truth class label (Point-wise color). | 3x 8-bit integers |
| **`ego_x, ego_y, ego_z`** | Cartesian coordinates of the vehicle. | Centimeters |
| **`ego_yaw`** | Yaw angle of the vehicle (Z-axis rotation). | Hundredth of degree |

### 3. Class Label Mapping (RGB)
The `r, g, b` fields correspond to specific object classes:

| Class ID | Label | R | G | B |
| :---: | :--- | :---: | :---: | :---: |
| **0** | **Antenna** | 38 | 23 | 180 |
| **1** | **Cable** | 177 | 132 | 47 |
| **2** | **Electric pole** | 129 | 81 | 97 |
| **3** | **Wind turbine** | 66 | 132 | 9 |

### 4. Frame Identification
To process the data frame by frame, you must filter points based on the Ego vehicle's pose.
The quadruplet **`(ego_x, ego_y, ego_z, ego_yaw)`** is the unique identifier for a single frame.

<table style="width:100%; text-align:center;">
  <tr>
    <td style="width:25%;">
      <img src="images/angle_side_view.png" style="width:100%;">
      <br>
      <sup>Angles definition, side view.</sup>
    </td>
    <td style="width:15%;">
      <img src="images/angle_top_view.png" style="width:100%;">
      <br>
      <sup>Angles definition, top view.</sup>
    </td>
  </tr>
</table>

---

## 📦 Deliverables & Expected Outputs

On the final day (D-7), the evaluation dataset will be sent between 00:01 and 01:00 AM. You will receive a total of 8 HDF5 files. These files correspond to two distinct scences with 100 frames each.
- **Scene A (Known)**: New frames from an environment you encountered during training.
- **Scene B (Unknown)**: A completely new environment.

For each scene, we provide 4 separate files containing the same frames but with different point densities. They will be identified by the following suffixes:
- `_100.h5`: Full resolution (100% of points)
- `_75.h5`: Downsampled to 75%
- `_50.h5`: Downsampled to 50%
- `_25.h5`: Downsampled to 25%

> **Note:** These evaluation files will contain RGB labels all set to 128. You must process these raw point clouds using your algorithm to generate the required prediction CSVs.

### Deliverables

1.  **Detection Algorithm/Model:** Stored in a standard format (preferred: **ONNX** or **PyTorch**).
2.  **Train Code:** Python source code to train the model (must include a `requirements.txt`).
3.  **Inference Code:** Python source code to run the model (must include a `requirements.txt`).
4.  **Visualization:** Screenshots of **max 10 frames** showing point clouds with predicted 3D bounding boxes colored by class.
5. **Prediction CSVs:**
    * 8 CSV files containing predictions on the evaluation datasets (see format below) but using **100%**, **75%**, **50%**, and **25%** of the input points.
    * Filenames should clearly identify the original file and point proportion used.
6. **Presentation:** A 2-minute video with the team presentation and a showcase of your solution with a quick explanation for a non-expert audience + a one-pager (doc or slide) with the models tested, the choices made during training and so on. Make sure to include the **parameters count** of your final model.

### CSV Output Format
Your inference script must output a CSV file with these exact columns:

* `ego_x`, `ego_y`, `ego_z`, `ego_yaw` (should be copied without modification from the input file to identify the frame)
* `bbox_center_x`, `bbox_center_y`, `bbox_center_z` (coordinates of the center of the detected bounding box in meters ; the origin of the center_x, center_y, center_z coordinates is the position of the Lidar (left-handed convention, z up))
* `bbox_width`, `bbox_length`, `bbox_height` (length of the bounding box along x, y and z axis in m before applying the yaw angle to the bounding box)
* `bbox_yaw` (rotation around Z axis)
* `class_ID` (0, 1, 2, or 3)
* `class_label` (Antenna, Cable, Electric Pole, Wind Turbine)


### 📩 Submission
Please send all deliverables by email to the following address:
**contact.hackathon-ai.ah@airbus.com**

---

## 🏆 Evaluation Criteria

Models will be evaluated based on the following metrics:

1.  **mAP @ IoU=0.5:** Mean Average Precision with an Intersection over Union threshold of 0.5.
2.  **Mean IoU (Correct Class):** The average IoU for bounding boxes that are correctly classified.
3.  **Robustness (Point Density):** High scores for models that maintain performance when point density decreases (100% → 25%).
4.  **Efficiency:** High scores for models with a **low number of parameters**.
5.  **Stability:** Consistent performance across the 2 provided evaluation datasets.

---

## 🛠️ Toolkit Usage

### Installation
```bash
pip install -r requirements.txt
```

### 1. Visualization Script (`visualize.py`)
Visualizes the point cloud in the World Frame (teleporting camera to Ego position).
```bash
python visualize.py --file <path_to_file.h5> --pose-index <N>
```
- `--pose-index N`: Visualizes the N-th frame in the file.
- **Modes:** Automatically switches between "Ground Truth RGB" (if labels exist) and "Reflectivity Intensity".

### 2. Core Library (`lidar_utils.py`)
Contains reusable functions for your training scripts:
- `load_h5_data`: Loads HDF5 to Pandas DataFrame.
- `spherical_to_local_cartesian`: Converts raw units (cm, 1/100 deg) to local Cartesian meters.

---
## 🚀 Ready for Takeoff!
You now have everything you need to start playing with this task! Be creative, curious, enjoy the challenge and aim for the top!\
We hope you will enjoy this competition as much as we look forward to discovering your solutions.\
We remain available throughout the entire week for any questions at the following email address: **contact.hackathon-ai.ah@airbus.com**

---
© Copyright Airbus Helicopters 2026 \
*This document and all information contained herein is the sole property of Airbus. No intellectual property rights are granted by the delivery of this document or the disclosure of its content. This document shall not be reproduced or disclosed to a third party without the expressed written consent of Airbus. This document and its content shall not be used for any purpose other than that for which it is supplied.
Airbus, its logo and product names are registered trademarks.*
