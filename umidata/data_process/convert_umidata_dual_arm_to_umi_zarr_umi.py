#!/usr/bin/env python3
"""
Convert dual-arm UMI raw data into a dataset.zarr.zip file that can be read
directly by diffusion_policy.dataset.umi_dataset.UmiDataset.

Observed raw layout for each episode:
    episodeX/
      camera/color/pikaDepthCamera_l/*.jpg
      camera/color/pikaDepthCamera_r/*.jpg
      camera/color/pikaFisheyeCamera_l/*.jpg
      camera/color/pikaFisheyeCamera_r/*.jpg
      localization/pose/pika_l/*.json
      localization/pose/pika_r/*.json
      gripper/encoder/pika_l/*.json
      gripper/encoder/pika_r/*.json
      statistic.txt

When `sync.txt` exists and contains entries, the converter uses that ordering.
If `sync.txt` is missing or empty, it falls back to sorting payload files by
timestamp. Streams are aligned by nearest-neighbor lookup, with the left depth
camera timeline used as the reference timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UMI_PROJECT_ROOT = REPO_ROOT / "universal_manipulation_interface"
if str(UMI_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(UMI_PROJECT_ROOT))

CROP = True
FISHEYE = True
DATE = "20260610_shorts"
TRIM_START_SECONDS = 0.5
TRIM_END_SECONDS = 0.2
OUTPUT_IMAGE_SIZE = (224, 224)  # width, height

DEFAULT_INPUT_ROOT = REPO_ROOT / "umidata" / "double" / DATE
DEFAULT_TEST_OUTPUT_IMAGE = (
    REPO_ROOT / "umidata" / "data_process" / "resize_img" / "test_resize_output_dual.jpg"
)

LEFT_POSE_REL_PATH = Path("localization/pose/pika_l")
RIGHT_POSE_REL_PATH = Path("localization/pose/pika_r")
LEFT_GRIPPER_REL_PATH = Path("gripper/encoder/pika_l")
RIGHT_GRIPPER_REL_PATH = Path("gripper/encoder/pika_r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dual-arm UMI raw data to dataset.zarr.zip"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Directory containing episode folders directly, or a higher-level directory "
            "that contains nested episode folders."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dataset.zarr.zip path. Defaults to "
            "dataset/double/<input_name>_<RGB|fisheye>_<croped|no_crop>_dual.zarr.zip"
        ),
    )
    parser.add_argument(
        "--test-resize-image",
        type=Path,
        default=None,
        help="Optional path to a single RGB image for testing resize output only.",
    )
    parser.add_argument(
        "--test-output-image",
        type=Path,
        default=DEFAULT_TEST_OUTPUT_IMAGE,
        help="Output path for the resized preview image when --test-resize-image is used.",
    )
    parser.add_argument(
        "--trim-start-seconds",
        type=float,
        default=TRIM_START_SECONDS,
        help="Trim this many seconds from the start of the common valid range.",
    )
    parser.add_argument(
        "--trim-end-seconds",
        type=float,
        default=TRIM_END_SECONDS,
        help="Trim this many seconds from the end of the common valid range.",
    )
    parser.add_argument(
        "--episode-list",
        type=Path,
        default=None,
        help="Optional text file listing episode directories relative to input root, one per line.",
    )
    parser.add_argument(
        "--max-allowed-gap",
        type=float,
        default=0.12,
        help="Maximum allowed alignment error in seconds for any non-reference stream.",
    )
    parser.add_argument(
        "--fisheye",
        action="store_true",
        default=FISHEYE,
        help="Use fisheye camera streams instead of depth camera streams.",
    )
    return parser.parse_args()


def episode_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.name
    suffix = stem.replace("episode", "")
    if suffix.isdigit():
        return (int(suffix), stem)
    return (10**9, stem)


def read_episode_list(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def get_camera_rel_paths(use_fisheye: bool) -> Tuple[Path, Path]:
    if use_fisheye:
        return (
            Path("camera/color/pikaFisheyeCamera_l"),
            Path("camera/color/pikaFisheyeCamera_r"),
        )
    return (
        Path("camera/color/pikaDepthCamera_l"),
        Path("camera/color/pikaDepthCamera_r"),
    )


def parse_timestamp_from_path(path: Path) -> float:
    return float(path.stem)


def collect_timestamped_files(directory: Path, suffix: str) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")

    files = sorted(directory.glob(f"*{suffix}"), key=parse_timestamp_from_path)
    if not files:
        raise FileNotFoundError(f"No `{suffix}` files found in: {directory}")
    return files


def resolve_stream_files(
    directory: Path,
    suffix: str,
    sync_file_name: str = "sync.txt",
) -> List[Path]:
    payload_files = collect_timestamped_files(directory, suffix)
    payload_by_name = {path.name: path for path in payload_files}

    sync_path = directory / sync_file_name
    if not sync_path.is_file() or sync_path.stat().st_size <= 0:
        return payload_files

    try:
        sync_names = read_nonempty_lines(sync_path)
    except Exception:
        return payload_files

    if not sync_names:
        return payload_files

    resolved_files: List[Path] = []
    missing_names: List[str] = []
    for name in sync_names:
        path = payload_by_name.get(name)
        if path is None:
            missing_names.append(name)
            continue
        resolved_files.append(path)

    if missing_names:
        preview = missing_names[:10]
        raise FileNotFoundError(
            f"Sync file references missing payloads in {directory}: {preview}"
        )
    if not resolved_files:
        raise RuntimeError(f"Sync file has no usable entries in {directory}")
    return resolved_files


def read_pose_json(json_path: Path) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    with json_path.open("r", encoding="utf-8") as f:
        pose_dict = json.load(f)

    xyz = np.array([pose_dict["x"], pose_dict["y"], pose_dict["z"]], dtype=np.float32)
    rpy = np.array(
        [pose_dict["roll"], pose_dict["pitch"], pose_dict["yaw"]],
        dtype=np.float32,
    )
    rotvec = Rotation.from_euler("xyz", rpy, degrees=False).as_rotvec().astype(
        np.float32
    )
    return np.concatenate([xyz, rotvec], axis=0)


def read_gripper_width(json_path: Path) -> float:
    with json_path.open("r", encoding="utf-8") as f:
        gripper_dict = json.load(f)
    return float(gripper_dict["distance"])


def read_rgb_image(image_path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if CROP:
        height, width = image_rgb.shape[:2]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        cropped = image_rgb[top : top + crop_size, left : left + crop_size]
        resized = cv2.resize(cropped, OUTPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(
            image_rgb, OUTPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA
        )

    return resized.astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), image_bgr)
    if not success:
        raise RuntimeError(f"Failed to save image to: {output_path}")


def run_resize_image_test(input_image: Path, output_image: Path) -> None:
    input_image = input_image.expanduser().resolve()
    output_image = output_image.expanduser().resolve()
    if not input_image.is_file():
        raise FileNotFoundError(f"Test image does not exist: {input_image}")

    resized_rgb = read_rgb_image(input_image)
    save_rgb_image(resized_rgb, output_image)

    print(f"Saved resized preview image to: {output_image}")
    print(f"Resized image shape: {resized_rgb.shape}")
    print(f"Resized image dtype: {resized_rgb.dtype}")


def nearest_indices(
    reference_timestamps: np.ndarray, candidate_timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    insert_idx = np.searchsorted(candidate_timestamps, reference_timestamps, side="left")
    left_idx = np.clip(insert_idx - 1, 0, len(candidate_timestamps) - 1)
    right_idx = np.clip(insert_idx, 0, len(candidate_timestamps) - 1)

    left_delta = np.abs(candidate_timestamps[left_idx] - reference_timestamps)
    right_delta = np.abs(candidate_timestamps[right_idx] - reference_timestamps)
    use_right = right_delta < left_delta
    best_idx = np.where(use_right, right_idx, left_idx)
    best_delta = np.where(use_right, right_delta, left_delta)
    return best_idx, best_delta


def build_shifted_action(
    pose0: np.ndarray,
    grip0: np.ndarray,
    pose1: np.ndarray,
    grip1: np.ndarray,
) -> np.ndarray:
    next_pose0 = np.concatenate([pose0[1:], pose0[-1:]], axis=0)
    next_grip0 = np.concatenate([grip0[1:], grip0[-1:]], axis=0)
    next_pose1 = np.concatenate([pose1[1:], pose1[-1:]], axis=0)
    next_grip1 = np.concatenate([grip1[1:], grip1[-1:]], axis=0)
    return np.concatenate([next_pose0, next_grip0, next_pose1, next_grip1], axis=-1)


def ensure_non_empty_streams(streams: Sequence[Tuple[str, List[Path]]]) -> None:
    for name, files in streams:
        if not files:
            raise ValueError(f"Stream `{name}` has no usable files.")


def build_episode_data(
    episode_dir: Path,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float = 0.0,
    max_allowed_gap: float | None = None,
    use_fisheye: bool = FISHEYE,
) -> dict:
    left_camera_rel_path, right_camera_rel_path = get_camera_rel_paths(use_fisheye)
    left_camera_files = resolve_stream_files(
        episode_dir / left_camera_rel_path, ".jpg"
    )
    right_camera_files = resolve_stream_files(
        episode_dir / right_camera_rel_path, ".jpg"
    )
    left_pose_files = resolve_stream_files(
        episode_dir / LEFT_POSE_REL_PATH, ".json"
    )
    right_pose_files = resolve_stream_files(
        episode_dir / RIGHT_POSE_REL_PATH, ".json"
    )
    left_gripper_files = resolve_stream_files(
        episode_dir / LEFT_GRIPPER_REL_PATH, ".json"
    )
    right_gripper_files = resolve_stream_files(
        episode_dir / RIGHT_GRIPPER_REL_PATH, ".json"
    )
    ensure_non_empty_streams(
        [
            ("camera_l", left_camera_files),
            ("camera_r", right_camera_files),
            ("pose_l", left_pose_files),
            ("pose_r", right_pose_files),
            ("gripper_l", left_gripper_files),
            ("gripper_r", right_gripper_files),
        ]
    )

    left_camera_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in left_camera_files], dtype=np.float64
    )
    right_camera_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in right_camera_files], dtype=np.float64
    )
    left_pose_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in left_pose_files], dtype=np.float64
    )
    right_pose_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in right_pose_files], dtype=np.float64
    )
    left_gripper_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in left_gripper_files], dtype=np.float64
    )
    right_gripper_ts = np.asarray(
        [parse_timestamp_from_path(path) for path in right_gripper_files], dtype=np.float64
    )

    common_start = max(
        left_camera_ts[0],
        right_camera_ts[0],
        left_pose_ts[0],
        right_pose_ts[0],
        left_gripper_ts[0],
        right_gripper_ts[0],
    )
    common_end = min(
        left_camera_ts[-1],
        right_camera_ts[-1],
        left_pose_ts[-1],
        right_pose_ts[-1],
        left_gripper_ts[-1],
        right_gripper_ts[-1],
    )

    valid_start = common_start + trim_start_seconds
    valid_end = common_end - trim_end_seconds
    if valid_start >= valid_end:
        raise ValueError(
            "Trim range removes the whole common interval. "
            f"common=[{common_start:.6f}, {common_end:.6f}] "
            f"trim=({trim_start_seconds:.3f}, {trim_end_seconds:.3f})."
        )

    reference_mask = (left_camera_ts >= valid_start) & (left_camera_ts <= valid_end)
    if not np.any(reference_mask):
        raise ValueError("No left-camera frames remain in the valid time range.")

    reference_files = [path for path, keep in zip(left_camera_files, reference_mask) if keep]
    reference_ts = left_camera_ts[reference_mask]

    right_cam_idx, right_cam_delta = nearest_indices(reference_ts, right_camera_ts)
    left_pose_idx, left_pose_delta = nearest_indices(reference_ts, left_pose_ts)
    right_pose_idx, right_pose_delta = nearest_indices(reference_ts, right_pose_ts)
    left_grip_idx, left_grip_delta = nearest_indices(reference_ts, left_gripper_ts)
    right_grip_idx, right_grip_delta = nearest_indices(reference_ts, right_gripper_ts)

    non_ref_deltas = {
        "camera_r": right_cam_delta,
        "pose_l": left_pose_delta,
        "pose_r": right_pose_delta,
        "gripper_l": left_grip_delta,
        "gripper_r": right_grip_delta,
    }
    if max_allowed_gap is not None:
        too_large = [
            f"{name}: max_delta={float(np.max(delta)):.6f}s"
            for name, delta in non_ref_deltas.items()
            if len(delta) > 0 and float(np.max(delta)) > max_allowed_gap
        ]
        if too_large:
            raise ValueError(
                "Alignment gap is too large for one or more streams: "
                + ", ".join(too_large)
            )

    camera0_rgb = []
    camera1_rgb = []
    robot0_pose = []
    robot1_pose = []
    robot0_gripper = []
    robot1_gripper = []

    for ref_i, image_path in enumerate(reference_files):
        camera0_rgb.append(read_rgb_image(image_path))
        camera1_rgb.append(read_rgb_image(right_camera_files[int(right_cam_idx[ref_i])]))
        robot0_pose.append(read_pose_json(left_pose_files[int(left_pose_idx[ref_i])]))
        robot1_pose.append(read_pose_json(right_pose_files[int(right_pose_idx[ref_i])]))
        robot0_gripper.append(
            read_gripper_width(left_gripper_files[int(left_grip_idx[ref_i])])
        )
        robot1_gripper.append(
            read_gripper_width(right_gripper_files[int(right_grip_idx[ref_i])])
        )

    robot0_pose = np.stack(robot0_pose, axis=0).astype(np.float32)
    robot1_pose = np.stack(robot1_pose, axis=0).astype(np.float32)
    camera0_rgb = np.stack(camera0_rgb, axis=0).astype(np.uint8)
    camera1_rgb = np.stack(camera1_rgb, axis=0).astype(np.uint8)
    robot0_gripper = np.asarray(robot0_gripper, dtype=np.float32).reshape(-1, 1)
    robot1_gripper = np.asarray(robot1_gripper, dtype=np.float32).reshape(-1, 1)

    seq_len = robot0_pose.shape[0]
    robot0_start_pose = np.repeat(robot0_pose[:1], repeats=seq_len, axis=0)
    robot0_end_pose = np.repeat(robot0_pose[-1:], repeats=seq_len, axis=0)
    robot1_start_pose = np.repeat(robot1_pose[:1], repeats=seq_len, axis=0)
    robot1_end_pose = np.repeat(robot1_pose[-1:], repeats=seq_len, axis=0)
    action = build_shifted_action(
        pose0=robot0_pose,
        grip0=robot0_gripper,
        pose1=robot1_pose,
        grip1=robot1_gripper,
    ).astype(np.float32)

    return {
        "camera0_rgb": camera0_rgb,
        "camera1_rgb": camera1_rgb,
        "robot0_eef_pos": robot0_pose[:, :3],
        "robot0_eef_rot_axis_angle": robot0_pose[:, 3:],
        "robot0_gripper_width": robot0_gripper,
        "robot1_eef_pos": robot1_pose[:, :3],
        "robot1_eef_rot_axis_angle": robot1_pose[:, 3:],
        "robot1_gripper_width": robot1_gripper,
        "robot0_demo_start_pose": robot0_start_pose,
        "robot0_demo_end_pose": robot0_end_pose,
        "robot1_demo_start_pose": robot1_start_pose,
        "robot1_demo_end_pose": robot1_end_pose,
        "action": action,
    }


def collect_episode_dirs(input_root: Path, episode_list_path: Path | None = None) -> List[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if episode_list_path is not None:
        episode_list_path = episode_list_path.expanduser().resolve()
        if not episode_list_path.is_file():
            raise FileNotFoundError(f"Episode list does not exist: {episode_list_path}")
        episode_names = read_episode_list(episode_list_path)
        if not episode_names:
            raise RuntimeError(f"Episode list is empty: {episode_list_path}")
        episode_dirs = [input_root / name for name in episode_names]
        missing_episode_dirs = [path for path in episode_dirs if not path.is_dir()]
        if missing_episode_dirs:
            missing_preview = [str(path) for path in missing_episode_dirs[:10]]
            raise FileNotFoundError(
                f"Some episode directories do not exist under {input_root}: {missing_preview}"
            )
        return sorted(episode_dirs, key=episode_sort_key)

    direct_episode_dirs = sorted(
        [
            path
            for path in input_root.iterdir()
            if path.is_dir() and path.name.startswith("episode")
        ],
        key=episode_sort_key,
    )
    if direct_episode_dirs:
        return direct_episode_dirs

    nested_episode_dirs = sorted(
        [path for path in input_root.rglob("episode*") if path.is_dir()],
        key=lambda path: (str(path.parent),) + episode_sort_key(path),
    )
    if nested_episode_dirs:
        return nested_episode_dirs

    raise RuntimeError(f"No episode directories found in {input_root}")


def resolve_output_path(
    input_root: Path, output: Path | None, use_fisheye: bool = FISHEYE
) -> Path:
    if output is not None:
        return output.expanduser().resolve()
    return (
        REPO_ROOT
        / "dataset"
        / "double"
        / (
            f"{input_root.name}_{'fisheye' if use_fisheye else 'RGB'}_"
            f"{'croped' if CROP else 'no_crop'}_dual.zarr.zip"
        )
    )


def main() -> None:
    args = parse_args()
    if args.test_resize_image is not None:
        run_resize_image_test(
            input_image=args.test_resize_image,
            output_image=args.test_output_image,
        )
        return

    input_root = args.input_root.expanduser().resolve()
    output_path = resolve_output_path(
        input_root=input_root,
        output=args.output,
        use_fisheye=args.fisheye,
    )

    import zarr
    from tqdm import tqdm

    from diffusion_policy.common.replay_buffer import ReplayBuffer

    episode_dirs = collect_episode_dirs(
        input_root=input_root,
        episode_list_path=args.episode_list,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    success_count = 0
    skipped = []

    print(f"Input root: {input_root}")
    print(f"Output path: {output_path}")
    print(f"Camera mode: {'fisheye' if args.fisheye else 'RGB'}")
    print(f"Trim start seconds: {args.trim_start_seconds}")
    print(f"Trim end seconds: {args.trim_end_seconds}")
    print(f"Max allowed gap: {args.max_allowed_gap}")
    print(f"Found {len(episode_dirs)} episode directories.")

    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        try:
            episode_data = build_episode_data(
                episode_dir=episode_dir,
                trim_start_seconds=args.trim_start_seconds,
                trim_end_seconds=args.trim_end_seconds,
                max_allowed_gap=args.max_allowed_gap,
                use_fisheye=args.fisheye,
            )
            replay_buffer.add_episode(data=episode_data, compressors=None)
            success_count += 1
        except Exception as exc:
            skipped.append((str(episode_dir), str(exc)))
            print(f"[SKIP] {episode_dir}: {exc}")

    if success_count == 0:
        raise RuntimeError("No episodes were converted successfully.")

    if output_path.exists():
        output_path.unlink()

    with zarr.ZipStore(str(output_path), mode="w") as zip_store:
        replay_buffer.save_to_store(store=zip_store)

    print(f"Converted {success_count} episodes.")
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Saved dataset to: {output_path}")

    if skipped:
        print("Skipped episodes:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
