#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# Keep these transforms aligned with calibration/tag_pose_transform.py.
T_BASE_TAG = np.array([-0.54569903, 0.55935591, -0.08874382], dtype=np.float64)
R_BASE_TAG = np.array(
    [
        [-0.79733017, 0.60348766, 0.00820007],
        [-0.60345291, -0.79690186, -0.02814284],
        [-0.01044921, -0.02738749, 0.99957028],
    ],
    dtype=np.float64,
)

T_SENSOR_CAM = np.array([-0.18459969, -0.01387259, 0.04480647], dtype=np.float64)
R_SENSOR_CAM = np.array(
    [
        [-0.00000457, 0.00487366, 0.99998812],
        [-0.99999507, 0.00313843, -0.00001987],
        [-0.00313849, -0.99998320, 0.00487363],
    ],
    dtype=np.float64,
)

# Rotate the camera frame so camera Y becomes umi_l X.
T_CAMERA_UMI = np.array([0.0, 0.0, 0.0], dtype=np.float64)
R_CAMERA_UMI = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert arm sensor poses in base_link to umi poses in tag frame "
            "in-place for all episodes under the target root."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/sunpeng/sp/umi_project/umidata/single/20260601"),
        help="Root directory containing episode*/arm/endPose/sensorPose.",
    )
    return parser.parse_args()


def compose_transform(rot_a_b, trans_a_b, rot_b_c, trans_b_c):
    rot_a_c = rot_a_b @ rot_b_c
    trans_a_c = rot_a_b @ trans_b_c + trans_a_b
    return rot_a_c, trans_a_c


def invert_transform(rot_a_b, trans_a_b):
    rot_b_a = rot_a_b.T
    trans_b_a = -rot_b_a @ trans_a_b
    return rot_b_a, trans_b_a


def load_sensor_pose(sensor_pose_path: Path):
    with sensor_pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)
    rot_base_sensor = Rotation.from_euler(
        "xyz",
        [pose["roll"], pose["pitch"], pose["yaw"]],
        degrees=False,
    ).as_matrix()
    trans_base_sensor = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    return pose, rot_base_sensor, trans_base_sensor


def convert_sensor_base_to_umi_tag(rot_base_sensor, trans_base_sensor):
    rot_base_cam, trans_base_cam = compose_transform(
        rot_base_sensor, trans_base_sensor, R_SENSOR_CAM, T_SENSOR_CAM
    )
    rot_base_umi, trans_base_umi = compose_transform(
        rot_base_cam, trans_base_cam, R_CAMERA_UMI, T_CAMERA_UMI
    )
    rot_tag_base, trans_tag_base = invert_transform(R_BASE_TAG, T_BASE_TAG)
    rot_tag_umi, trans_tag_umi = compose_transform(
        rot_tag_base, trans_tag_base, rot_base_umi, trans_base_umi
    )
    return rot_tag_umi, trans_tag_umi


def build_updated_pose_payload(source_pose, rot_tag_umi, trans_tag_umi):
    rot_obj = Rotation.from_matrix(rot_tag_umi)
    roll, pitch, yaw = rot_obj.as_euler("xyz", degrees=False)
    updated_pose = dict(source_pose)
    updated_pose["x"] = float(trans_tag_umi[0])
    updated_pose["y"] = float(trans_tag_umi[1])
    updated_pose["z"] = float(trans_tag_umi[2])
    updated_pose["roll"] = float(roll)
    updated_pose["pitch"] = float(pitch)
    updated_pose["yaw"] = float(yaw)
    return updated_pose


def convert_episode_inplace(sensor_pose_dir: Path):
    converted = 0
    sensor_pose_paths = sorted(sensor_pose_dir.glob("*.json"))
    for sensor_pose_path in sensor_pose_paths:
        source_pose, rot_base_sensor, trans_base_sensor = load_sensor_pose(sensor_pose_path)
        rot_tag_umi, trans_tag_umi = convert_sensor_base_to_umi_tag(
            rot_base_sensor, trans_base_sensor
        )
        payload = build_updated_pose_payload(source_pose, rot_tag_umi, trans_tag_umi)
        with sensor_pose_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        converted += 1

    return {
        "sensor_pose_dir": str(sensor_pose_dir),
        "n_total": len(sensor_pose_paths),
        "n_converted": converted,
    }


def find_sensor_pose_dirs(input_root: Path):
    return sorted(input_root.glob("episode*/arm/endPose/sensorPose"))


def main():
    args = parse_args()
    sensor_pose_dirs = find_sensor_pose_dirs(args.input_root)
    if not sensor_pose_dirs:
        raise RuntimeError(f"No episode*/arm/endPose/sensorPose found under {args.input_root}")

    summaries = []
    total_converted = 0
    total_files = 0
    for sensor_pose_dir in sensor_pose_dirs:
        summary = convert_episode_inplace(sensor_pose_dir=sensor_pose_dir)
        summaries.append(summary)
        total_converted += summary["n_converted"]
        total_files += summary["n_total"]
        print(
            f"{sensor_pose_dir.parent.parent.parent.name}: "
            f"{summary['n_converted']}/{summary['n_total']} updated in-place "
            f"-> {summary['sensor_pose_dir']}"
        )

    print(
        f"Finished converting {total_converted}/{total_files} files across "
        f"{len(summaries)} episodes."
    )


if __name__ == "__main__":
    main()
