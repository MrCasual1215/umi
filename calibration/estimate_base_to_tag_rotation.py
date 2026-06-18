#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ARUCO_DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]

# Fixed 2x3 board layout specified by the user:
# 5 0 1
# 3 2 4
#
# Board frame:
# - origin at marker 4 center
# - +x points from marker 4 center to marker 1 center
# - +y points from marker 4 center to marker 2 center
# - +z follows the right-hand rule
BOARD_CENTER_GRID = {
    4: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, 2),
    0: (1, 1),
    5: (1, 2),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate T_base_board and T_sensor_cam from a fixed 2x3 ArUco board."
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("/home/sunpeng/sp/umi_project/calibration/data/episode0"),
        help="Episode directory containing camera and sensorPose data.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="pikaGripperDepthCamera",
        help="Camera folder name under camera/color/.",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="auto",
        help="ArUco dictionary name such as DICT_4X4_250, or 'auto'.",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=0.0684,
        help="Marker side length in meters.",
    )
    parser.add_argument(
        "--marker-gap-x",
        type=float,
        default=0.0074,
        help="Horizontal edge-to-edge marker gap in meters.",
    )
    parser.add_argument(
        "--marker-gap-y",
        type=float,
        default=0.0100,
        help="Vertical edge-to-edge marker gap in meters.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.05,
        help="Maximum allowed timestamp mismatch between image and sensor pose in seconds.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images to process.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Process every Nth image.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "JSON output path. If omitted, defaults to "
            "<episode-dir>/base_to_board_calibration.json."
        ),
    )
    parser.add_argument(
        "--min-visible-tags",
        type=int,
        default=4,
        help="Require at least this many board tags to be visible in a frame.",
    )
    parser.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=2.5,
        help="Reject frame poses whose mean reprojection error exceeds this threshold.",
    )
    parser.add_argument(
        "--handeye-method",
        type=str,
        default="auto",
        help=(
            "OpenCV hand-eye method: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS, "
            "or auto to try all available methods."
        ),
    )
    return parser.parse_args()


def load_camera_intrinsics(camera_config_path: Path):
    with camera_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    k = np.array(cfg["K"], dtype=np.float64).reshape(3, 3)
    d = np.array(cfg["D"], dtype=np.float64).reshape(-1, 1)
    return k, d


def load_sensor_pose(sensor_pose_path: Path, euler_convention: str):
    with sensor_pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)
    rot = Rotation.from_euler(
        euler_convention,
        [pose["roll"], pose["pitch"], pose["yaw"]],
        degrees=False,
    ).as_matrix()
    pos = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    return rot, pos


def list_timestamped_files(directory: Path, suffix: str):
    files = sorted(directory.glob(f"*{suffix}"))
    timestamps = np.array([float(path.stem) for path in files], dtype=np.float64)
    return files, timestamps


def get_aruco_dict(dict_name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def detect_markers(image_bgr, aruco_dict):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(
            image_bgr, aruco_dict, parameters=params
        )
    elif hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image_bgr)
    else:
        raise RuntimeError("This OpenCV build does not provide a supported ArUco detector API.")
    if ids is None:
        return []
    return [(int(tag_id[0]), corner) for tag_id, corner in zip(ids, corners)]


def auto_select_aruco_dict(image_paths):
    sample_paths = image_paths[: min(len(image_paths), 30)]
    best_dict = None
    best_score = -1
    for dict_name in ARUCO_DICT_NAMES:
        aruco_dict = get_aruco_dict(dict_name)
        score = 0
        for image_path in sample_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            score += len(detect_markers(image, aruco_dict))
        if score > best_score:
            best_dict = dict_name
            best_score = score
    if best_dict is None or best_score <= 0:
        raise RuntimeError("Failed to detect any ArUco marker with common dictionaries.")
    return best_dict


def build_board_object_points(marker_length, marker_gap_x, marker_gap_y):
    half = marker_length / 2.0
    center_step_x = marker_length + marker_gap_x
    center_step_y = marker_length + marker_gap_y
    object_points = {}
    for marker_id, (grid_x, grid_y) in BOARD_CENTER_GRID.items():
        center_x = grid_x * center_step_x
        center_y = grid_y * center_step_y
        object_points[marker_id] = np.array(
            [
                [center_x + half, center_y + half, 0.0],
                [center_x + half, center_y - half, 0.0],
                [center_x - half, center_y - half, 0.0],
                [center_x - half, center_y + half, 0.0],
            ],
            dtype=np.float32,
        )
    return object_points


def estimate_board_pose(
    image_bgr,
    aruco_dict,
    camera_matrix,
    dist_coeffs,
    board_object_points,
):
    detections = detect_markers(image_bgr, aruco_dict)
    object_points = []
    image_points = []
    visible_ids = []
    for detected_tag_id, corners in detections:
        if detected_tag_id not in board_object_points:
            continue
        visible_ids.append(detected_tag_id)
        object_points.append(board_object_points[detected_tag_id])
        image_points.append(corners.reshape(-1, 2).astype(np.float32))

    if len(visible_ids) == 0:
        return {
            "r_cam_board": None,
            "t_cam_board": None,
            "visible_ids": [],
            "n_visible_tags": 0,
            "reproj_error_px": None,
        }

    object_points = np.concatenate(object_points, axis=0)
    image_points = np.concatenate(image_points, axis=0)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {
            "r_cam_board": None,
            "t_cam_board": None,
            "visible_ids": sorted(visible_ids),
            "n_visible_tags": int(len(visible_ids)),
            "reproj_error_px": None,
        }

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )

    projected_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    reproj_error_px = float(
        np.mean(np.linalg.norm(projected_points - image_points, axis=1))
    )
    return {
        "r_cam_board": Rotation.from_rotvec(rvec.reshape(3)).as_matrix(),
        "t_cam_board": tvec.reshape(3).astype(np.float64),
        "visible_ids": sorted(visible_ids),
        "n_visible_tags": int(len(visible_ids)),
        "reproj_error_px": reproj_error_px,
    }


def build_sensor_pose_interpolators(sensor_timestamps, sensor_rotations, sensor_positions):
    if len(sensor_timestamps) < 2:
        raise RuntimeError("At least two sensor poses are required for interpolation.")
    rot_obj = Rotation.from_matrix(np.stack(sensor_rotations, axis=0))
    pos_arr = np.stack(sensor_positions, axis=0)
    return Slerp(sensor_timestamps, rot_obj), pos_arr


def interpolate_sensor_pose(
    image_timestamp, sensor_timestamps, sensor_slerp, sensor_positions, max_time_diff
):
    insert_idx = int(np.searchsorted(sensor_timestamps, image_timestamp))
    if insert_idx <= 0:
        left_idx = right_idx = 0
    elif insert_idx >= len(sensor_timestamps):
        left_idx = right_idx = len(sensor_timestamps) - 1
    else:
        left_idx = insert_idx - 1
        right_idx = insert_idx

    left_dt = abs(image_timestamp - sensor_timestamps[left_idx])
    right_dt = abs(sensor_timestamps[right_idx] - image_timestamp)
    nearest_dt = min(left_dt, right_dt)
    if nearest_dt > max_time_diff:
        return None, nearest_dt, left_idx, right_idx

    if left_idx == right_idx:
        rot = sensor_slerp([sensor_timestamps[left_idx]]).as_matrix()[0]
        pos = sensor_positions[left_idx]
        return (rot, pos), nearest_dt, left_idx, right_idx

    rot = sensor_slerp([image_timestamp]).as_matrix()[0]
    pos = np.array(
        [
            np.interp(image_timestamp, sensor_timestamps, sensor_positions[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    return (rot, pos), nearest_dt, left_idx, right_idx


def average_rotations(rotation_mats):
    rot_obj = Rotation.from_matrix(np.stack(rotation_mats, axis=0))
    try:
        return rot_obj.mean().as_matrix()
    except Exception:
        mean_matrix = np.mean(rot_obj.as_matrix(), axis=0)
        u, _, vt = np.linalg.svd(mean_matrix)
        mean_rot = u @ vt
        if np.linalg.det(mean_rot) < 0:
            u[:, -1] *= -1
            mean_rot = u @ vt
        return mean_rot


def rotation_errors_deg(rotation_mats, reference_rot):
    ref = Rotation.from_matrix(reference_rot)
    errors = []
    for rot in rotation_mats:
        delta = ref.inv() * Rotation.from_matrix(rot)
        errors.append(np.degrees(np.linalg.norm(delta.as_rotvec())))
    return np.array(errors, dtype=np.float64)


def translation_errors_m(translation_vecs, reference_translation):
    reference_translation = np.asarray(reference_translation, dtype=np.float64).reshape(1, 3)
    translation_vecs = np.asarray(translation_vecs, dtype=np.float64)
    return np.linalg.norm(translation_vecs - reference_translation, axis=1)


def compose_transform(rot_a_b, trans_a_b, rot_b_c, trans_b_c):
    rot_a_c = rot_a_b @ rot_b_c
    trans_a_c = rot_a_b @ trans_b_c + trans_a_b
    return rot_a_c, trans_a_c


def format_python_value(value):
    return repr(float(value))


def format_python_vector(name, values):
    formatted = ", ".join(format_python_value(v) for v in values)
    return f"{name} = [{formatted}]"


def format_python_matrix(name, matrix):
    rows = []
    for row in matrix:
        formatted_row = ", ".join(format_python_value(v) for v in row)
        rows.append(f"    [{formatted_row}]")
    return f"{name} = [\n" + ",\n".join(rows) + "\n]"


def write_python_constants(py_out_path, estimated_solution):
    content = "\n\n".join(
        [
            format_python_vector(
                "T_BASE_TAG", estimated_solution["translation_base_board_m"]
            ),
            format_python_matrix(
                "R_BASE_TAG", estimated_solution["rotation_base_board"]
            ),
            format_python_vector(
                "T_SENSOR_CAM", estimated_solution["translation_sensor_cam_m"]
            ),
            format_python_matrix(
                "R_SENSOR_CAM", estimated_solution["rotation_sensor_cam"]
            ),
        ]
    )
    py_out_path.write_text(content + "\n", encoding="utf-8")


def compute_base_board_poses(
    sensor_rotations,
    sensor_positions,
    cam_board_rotations,
    cam_board_positions,
    r_sensor_cam,
    t_sensor_cam,
):
    base_board_rotations = []
    base_board_positions = []
    for r_base_sensor, t_base_sensor, r_cam_board, t_cam_board in zip(
        sensor_rotations,
        sensor_positions,
        cam_board_rotations,
        cam_board_positions,
    ):
        r_base_cam, t_base_cam = compose_transform(
            r_base_sensor, t_base_sensor, r_sensor_cam, t_sensor_cam
        )
        r_base_board, t_base_board = compose_transform(
            r_base_cam, t_base_cam, r_cam_board, t_cam_board
        )
        base_board_rotations.append(r_base_board)
        base_board_positions.append(t_base_board)
    return base_board_rotations, base_board_positions


def summarize_solution(
    matched_sensor_rotations,
    matched_sensor_positions,
    matched_cam_board_rotations,
    matched_cam_board_positions,
    solution_name,
    r_sensor_cam,
    t_sensor_cam,
):
    per_frame_rotations, per_frame_positions = compute_base_board_poses(
        matched_sensor_rotations,
        matched_sensor_positions,
        matched_cam_board_rotations,
        matched_cam_board_positions,
        r_sensor_cam,
        t_sensor_cam,
    )

    r_mean = average_rotations(per_frame_rotations)
    t_mean = np.mean(np.stack(per_frame_positions, axis=0), axis=0)
    rot_errors_deg = rotation_errors_deg(per_frame_rotations, r_mean)
    trans_errors_m = translation_errors_m(per_frame_positions, t_mean)

    rot_thresh = max(1.0, np.percentile(rot_errors_deg, 80))
    trans_thresh = max(0.005, np.percentile(trans_errors_m, 80))
    inlier_mask = (rot_errors_deg <= rot_thresh) & (trans_errors_m <= trans_thresh)

    inlier_rotations = [
        rot for rot, keep in zip(per_frame_rotations, inlier_mask.tolist()) if keep
    ]
    inlier_positions = [
        pos for pos, keep in zip(per_frame_positions, inlier_mask.tolist()) if keep
    ]
    if len(inlier_rotations) >= 3:
        r_mean = average_rotations(inlier_rotations)
        t_mean = np.mean(np.stack(inlier_positions, axis=0), axis=0)
        rot_errors_deg = rotation_errors_deg(inlier_rotations, r_mean)
        trans_errors_m = translation_errors_m(inlier_positions, t_mean)
        used_count = len(inlier_rotations)
    else:
        used_count = len(per_frame_rotations)

    return {
        "name": solution_name,
        "rotation_sensor_cam": np.round(r_sensor_cam, 8).tolist(),
        "translation_sensor_cam_m": np.round(t_sensor_cam, 8).tolist(),
        "rotation_base_board": np.round(r_mean, 8).tolist(),
        "translation_base_board_m": np.round(t_mean, 8).tolist(),
        "n_valid_pairs_before_filter": int(len(per_frame_rotations)),
        "n_used_for_final_estimate": int(used_count),
        "angular_error_deg_mean": float(np.mean(rot_errors_deg)),
        "angular_error_deg_median": float(np.median(rot_errors_deg)),
        "angular_error_deg_max": float(np.max(rot_errors_deg)),
        "translation_error_m_mean": float(np.mean(trans_errors_m)),
        "translation_error_m_median": float(np.median(trans_errors_m)),
        "translation_error_m_max": float(np.max(trans_errors_m)),
    }


def get_handeye_methods():
    methods = {}
    for name in ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"]:
        attr = f"CALIB_HAND_EYE_{name}"
        if hasattr(cv2, attr):
            methods[name] = getattr(cv2, attr)
    return methods


def run_handeye_calibration(
    method_name,
    sensor_rotations,
    sensor_positions,
    cam_board_rotations,
    cam_board_positions,
):
    methods = get_handeye_methods()
    if method_name not in methods:
        raise RuntimeError(f"Unsupported or unavailable hand-eye method: {method_name}")

    r_sensor_cam, t_sensor_cam = cv2.calibrateHandEye(
        R_gripper2base=[r.astype(np.float64) for r in sensor_rotations],
        t_gripper2base=[t.reshape(3, 1).astype(np.float64) for t in sensor_positions],
        R_target2cam=[r.astype(np.float64) for r in cam_board_rotations],
        t_target2cam=[t.reshape(3, 1).astype(np.float64) for t in cam_board_positions],
        method=methods[method_name],
    )
    return (
        np.asarray(r_sensor_cam, dtype=np.float64).reshape(3, 3),
        np.asarray(t_sensor_cam, dtype=np.float64).reshape(3),
    )


def estimate_sensor_cam(
    sensor_rotations,
    sensor_positions,
    cam_board_rotations,
    cam_board_positions,
    requested_method,
):
    available_methods = get_handeye_methods()
    if requested_method == "auto":
        method_names = list(available_methods.keys())
    else:
        method_names = [requested_method.upper()]

    candidates = []
    for method_name in method_names:
        try:
            r_sensor_cam, t_sensor_cam = run_handeye_calibration(
                method_name,
                sensor_rotations,
                sensor_positions,
                cam_board_rotations,
                cam_board_positions,
            )
            summary = summarize_solution(
                matched_sensor_rotations=sensor_rotations,
                matched_sensor_positions=sensor_positions,
                matched_cam_board_rotations=cam_board_rotations,
                matched_cam_board_positions=cam_board_positions,
                solution_name=f"handeye_{method_name}",
                r_sensor_cam=r_sensor_cam,
                t_sensor_cam=t_sensor_cam,
            )
            score = (
                summary["angular_error_deg_mean"]
                + 100.0 * summary["translation_error_m_mean"]
            )
            candidates.append(
                {
                    "method": method_name,
                    "score": float(score),
                    "summary": summary,
                }
            )
        except cv2.error:
            continue

    if not candidates:
        raise RuntimeError("All requested hand-eye calibration methods failed.")

    candidates.sort(key=lambda item: item["score"])
    return candidates[0], candidates


def evaluate_dataset(
    sensor_paths,
    sensor_timestamps,
    valid_detections,
    max_time_diff,
    handeye_method,
):
    euler_convention = "xyz"
    sensor_poses = [
        load_sensor_pose(path, euler_convention=euler_convention) for path in sensor_paths
    ]
    sensor_rotations = [rot for rot, _ in sensor_poses]
    sensor_positions = [pos for _, pos in sensor_poses]
    sensor_slerp, sensor_position_array = build_sensor_pose_interpolators(
        sensor_timestamps, sensor_rotations, sensor_positions
    )

    matched_sensor_rotations = []
    matched_sensor_positions = []
    matched_cam_board_rotations = []
    matched_cam_board_positions = []
    used_rows = []
    skipped_time = 0

    for row in valid_detections:
        image_timestamp = row["image_timestamp"]
        sensor_pose, dt, left_idx, right_idx = interpolate_sensor_pose(
            image_timestamp=image_timestamp,
            sensor_timestamps=sensor_timestamps,
            sensor_slerp=sensor_slerp,
            sensor_positions=sensor_position_array,
            max_time_diff=max_time_diff,
        )
        if sensor_pose is None:
            skipped_time += 1
            continue

        r_base_sensor, t_base_sensor = sensor_pose
        matched_sensor_rotations.append(r_base_sensor)
        matched_sensor_positions.append(t_base_sensor)
        matched_cam_board_rotations.append(row["r_cam_board"])
        matched_cam_board_positions.append(row["t_cam_board"])
        used_rows.append(
            {
                "image_timestamp": float(image_timestamp),
                "sensor_timestamp_left": float(sensor_timestamps[left_idx]),
                "sensor_timestamp_right": float(sensor_timestamps[right_idx]),
                "nearest_time_diff_sec": float(dt),
                "visible_ids": row["visible_ids"],
                "n_visible_tags": int(row["n_visible_tags"]),
                "reproj_error_px": float(row["reproj_error_px"]),
                "interpolated_sensor_position_m": np.round(t_base_sensor, 8).tolist(),
            }
        )

    if not matched_sensor_rotations:
        raise RuntimeError("No valid frame pairs found after timestamp alignment.")

    best_candidate, candidates = estimate_sensor_cam(
        sensor_rotations=matched_sensor_rotations,
        sensor_positions=matched_sensor_positions,
        cam_board_rotations=matched_cam_board_rotations,
        cam_board_positions=matched_cam_board_positions,
        requested_method=handeye_method,
    )

    return {
        "sensor_euler_convention": euler_convention,
        "n_valid_pairs": int(len(matched_sensor_rotations)),
        "skipped_time_mismatch": int(skipped_time),
        "handeye_method": best_candidate["method"],
        "handeye_candidates": [
            {
                "method": item["method"],
                "score": item["score"],
                "angular_error_deg_mean": item["summary"]["angular_error_deg_mean"],
                "translation_error_m_mean": item["summary"]["translation_error_m_mean"],
            }
            for item in candidates
        ],
        "estimated_solution": best_candidate["summary"],
        "used_rows_preview": used_rows[:5],
    }


def main():
    args = parse_args()
    if args.out is None:
        args.out = args.episode_dir / "base_to_board_calibration.json"
    py_out = args.out.with_suffix(".py")

    camera_dir = args.episode_dir / "camera" / "color" / args.camera_name
    sensor_dir = args.episode_dir / "arm" / "endPose" / "sensorPose"
    camera_cfg_path = camera_dir / "config.json"

    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    if not sensor_dir.is_dir():
        raise FileNotFoundError(f"Sensor pose directory not found: {sensor_dir}")
    if not camera_cfg_path.is_file():
        raise FileNotFoundError(f"Camera config not found: {camera_cfg_path}")

    camera_matrix, dist_coeffs = load_camera_intrinsics(camera_cfg_path)
    image_paths, image_timestamps = list_timestamped_files(camera_dir, ".jpg")
    sensor_paths, sensor_timestamps = list_timestamped_files(sensor_dir, ".json")

    if args.sample_step > 1:
        image_paths = image_paths[:: args.sample_step]
        image_timestamps = image_timestamps[:: args.sample_step]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        image_timestamps = image_timestamps[: args.max_images]
    if len(image_paths) == 0:
        raise RuntimeError("No images found to process.")
    if len(sensor_paths) == 0:
        raise RuntimeError("No sensor poses found to process.")

    aruco_dict_name = args.aruco_dict
    if aruco_dict_name == "auto":
        aruco_dict_name = auto_select_aruco_dict(image_paths)
    aruco_dict = get_aruco_dict(aruco_dict_name)
    board_object_points = build_board_object_points(
        marker_length=args.marker_length,
        marker_gap_x=args.marker_gap_x,
        marker_gap_y=args.marker_gap_y,
    )

    skipped_no_tag = 0
    skipped_high_reproj = 0
    skipped_too_few_tags = 0
    valid_detections = []

    for image_path, image_timestamp in zip(image_paths, image_timestamps):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        board_pose = estimate_board_pose(
            image_bgr=image,
            aruco_dict=aruco_dict,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            board_object_points=board_object_points,
        )
        if board_pose["n_visible_tags"] == 0:
            skipped_no_tag += 1
            continue
        if board_pose["n_visible_tags"] < args.min_visible_tags:
            skipped_too_few_tags += 1
            continue
        if board_pose["r_cam_board"] is None or board_pose["reproj_error_px"] is None:
            skipped_high_reproj += 1
            continue
        if board_pose["reproj_error_px"] > args.max_reproj_error_px:
            skipped_high_reproj += 1
            continue

        valid_detections.append(
            {
                "image_timestamp": float(image_timestamp),
                "r_cam_board": board_pose["r_cam_board"],
                "t_cam_board": board_pose["t_cam_board"],
                "visible_ids": board_pose["visible_ids"],
                "n_visible_tags": board_pose["n_visible_tags"],
                "reproj_error_px": board_pose["reproj_error_px"],
            }
        )

    if not valid_detections:
        raise RuntimeError(
            "No valid board detections found. Check the board ids, marker size/gap, and visibility."
        )

    evaluation = evaluate_dataset(
        sensor_paths=sensor_paths,
        sensor_timestamps=sensor_timestamps,
        valid_detections=valid_detections,
        max_time_diff=args.max_time_diff,
        handeye_method=args.handeye_method,
    )

    result = {
        "episode_dir": str(args.episode_dir),
        "camera_name": args.camera_name,
        "aruco_dict": aruco_dict_name,
        "board_marker_ids": sorted(BOARD_CENTER_GRID.keys()),
        "board_layout_rows": [[5, 0, 1], [3, 2, 4]],
        "board_frame_definition": {
            "origin": "marker_4_center",
            "x_axis": "marker_4_center_to_marker_1_center",
            "y_axis": "marker_4_center_to_marker_2_center",
            "z_axis": "right_hand_rule",
        },
        "marker_length": float(args.marker_length),
        "marker_gap_x": float(args.marker_gap_x),
        "marker_gap_y": float(args.marker_gap_y),
        "min_visible_tags": int(args.min_visible_tags),
        "max_reproj_error_px": float(args.max_reproj_error_px),
        "handeye_method_mode": args.handeye_method,
        "n_total_images": int(len(image_paths)),
        "n_valid_board_detections": int(len(valid_detections)),
        "skipped_no_tag": int(skipped_no_tag),
        "skipped_high_reproj_error": int(skipped_high_reproj),
        "skipped_too_few_tags": int(skipped_too_few_tags),
        "sensor_euler_convention": "xyz",
        "n_valid_pairs": evaluation["n_valid_pairs"],
        "skipped_time_mismatch": evaluation["skipped_time_mismatch"],
        "handeye_method": evaluation["handeye_method"],
        "handeye_candidates": evaluation["handeye_candidates"],
        "estimated_solution": evaluation["estimated_solution"],
        "used_rows_preview": evaluation["used_rows_preview"],
    }

    print("Sensor euler convention: xyz")
    print(f"Selected hand-eye method: {result['handeye_method']}")
    print("Estimated t_sensor_cam (m):")
    print(np.array(result["estimated_solution"]["translation_sensor_cam_m"]))
    print("Estimated R_sensor_cam:")
    print(np.array(result["estimated_solution"]["rotation_sensor_cam"]))
    print("Estimated t_base_board (m):")
    print(np.array(result["estimated_solution"]["translation_base_board_m"]))
    print("Estimated R_base_board:")
    print(np.array(result["estimated_solution"]["rotation_base_board"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_python_constants(py_out, result["estimated_solution"])
    print(f"Saved result to {args.out}")
    print(f"Saved python constants to {py_out}")


if __name__ == "__main__":
    main()
