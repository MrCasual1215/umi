#!/usr/bin/env python3
"""
Convert multiple single-arm UMI raw datasets into one merged dataset.zarr.zip.

The merged converter currently defaults to these date roots:
    - umidata/single/20260601
    - umidata/single/20260602
    - umidata/single/20260603

For each input root, the script auto-detects the pose layout:
    - arm/endPose/sensorPose
    - arm/endPose/gripperPose

Validation is intentionally simple:
    - required sync.txt files must exist
    - sync.txt entries must be non-empty
    - referenced payload files must exist
    - sequence length uses the shortest synced stream
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
UMI_PROJECT_ROOT = REPO_ROOT / "universal_manipulation_interface"
if str(UMI_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(UMI_PROJECT_ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


CROP = True
FISHEYE = False
DATE_NAMES = ("20260601", "20260602", "20260603")
TRIM_START_SECONDS = 0.1
TRIM_END_SECONDS = 0.2
DEFAULT_INPUT_ROOTS = [
    REPO_ROOT / "umidata" / "single" / date_name for date_name in DATE_NAMES
]

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "dataset"
    / "single"
    / f"{'_'.join(DATE_NAMES)}_{'fisheye' if FISHEYE else 'RGB'}_{'croped' if CROP else 'no_crop'}_merged.zarr.zip"
)
DEFAULT_TEST_OUTPUT_IMAGE = (
    REPO_ROOT / "umidata" / "data_process" / "resize_img" / "test_resize_output.jpg"
)

CAMERA_REL_PATH = Path(
    "camera/color/pikaGripperFisheyeCamera"
    if FISHEYE
    else "camera/color/pikaGripperDepthCamera"
)
GRIPPER_REL_PATH = Path("gripper/encoder/gripperWidth")


@dataclass(frozen=True)
class DateSpec:
    date: str
    input_root: Path
    pose_rel_path: Path
    episode_list_path: Path | None


POSE_REL_PATH_CANDIDATES = (
    Path("arm/endPose/sensorPose"),
    Path("arm/endPose/gripperPose"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multiple single-arm UMI raw datasets into one dataset.zarr.zip"
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUT_ROOTS,
        help="Input root directories to merge, each containing episode* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output merged dataset.zarr.zip path.",
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
        help="Trim this many seconds from the start of every episode.",
    )
    parser.add_argument(
        "--trim-end-seconds",
        type=float,
        default=TRIM_END_SECONDS,
        help="Trim this many seconds from the end of every episode.",
    )
    return parser.parse_args()


def episode_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.name
    suffix = stem.replace("episode", "")
    if suffix.isdigit():
        return (int(suffix), stem)
    return (10**9, stem)


def read_sync_file(sync_path: Path) -> List[str]:
    with sync_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    entries = [line for line in lines if line]
    if not entries:
        raise ValueError(f"sync.txt has no entries: {sync_path}")
    return entries


def read_episode_list(list_path: Path) -> List[str]:
    with list_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return [line for line in lines if line]


def parse_timestamp_from_path(file_name: str) -> float:
    return float(Path(file_name).stem)


def read_pose_json(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as file:
        pose_dict = json.load(file)

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
    with json_path.open("r", encoding="utf-8") as file:
        gripper_dict = json.load(file)
    return float(gripper_dict["distance"])


def read_rgb_image(image_path: Path) -> np.ndarray:
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
        resized = cv2.resize(
            cropped, (224, 224), interpolation=cv2.INTER_AREA
        )
    else:
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    return resized.astype(np.uint8)


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
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


def compute_trim_indices(
    camera_files: List[str],
    trim_start_seconds: float,
    trim_end_seconds: float,
) -> List[int]:
    if trim_start_seconds < 0 or trim_end_seconds < 0:
        raise ValueError("Trim seconds must be non-negative.")

    timestamps = np.asarray(
        [parse_timestamp_from_path(file_name) for file_name in camera_files],
        dtype=np.float64,
    )
    start_timestamp = timestamps[0] + trim_start_seconds
    end_timestamp = timestamps[-1] - trim_end_seconds
    if start_timestamp > end_timestamp:
        raise ValueError(
            "Trim range removes the whole episode. "
            f"start={trim_start_seconds:.3f}s, end={trim_end_seconds:.3f}s."
        )

    kept_indices = np.where(
        (timestamps >= start_timestamp) & (timestamps <= end_timestamp)
    )[0]
    if kept_indices.size == 0:
        raise ValueError(
            "No frames remain after trimming. "
            f"start={trim_start_seconds:.3f}s, end={trim_end_seconds:.3f}s."
        )
    return kept_indices.tolist()


def build_episode_data(
    episode_dir: Path,
    pose_rel_path: Path,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float = 0.0,
) -> dict:
    camera_dir = episode_dir / CAMERA_REL_PATH
    pose_dir = episode_dir / pose_rel_path
    gripper_dir = episode_dir / GRIPPER_REL_PATH

    camera_sync = camera_dir / "sync.txt"
    pose_sync = pose_dir / "sync.txt"
    gripper_sync = gripper_dir / "sync.txt"

    missing_paths = [
        str(path)
        for path in (camera_sync, pose_sync, gripper_sync)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Episode {episode_dir.name} is missing required sync files: {missing_paths}"
        )

    camera_files = read_sync_file(camera_sync)
    pose_files = read_sync_file(pose_sync)
    gripper_files = read_sync_file(gripper_sync)

    seq_len = min(len(camera_files), len(pose_files), len(gripper_files))
    if seq_len <= 0:
        raise ValueError(f"Episode {episode_dir.name} has no usable aligned frames.")

    if len({len(camera_files), len(pose_files), len(gripper_files)}) != 1:
        print(
            f"[WARN] {episode_dir.name}: sync length mismatch "
            f"(rgb={len(camera_files)}, pose={len(pose_files)}, gripper={len(gripper_files)}). "
            f"Using shortest length {seq_len}."
        )

    camera_files = camera_files[:seq_len]
    pose_files = pose_files[:seq_len]
    gripper_files = gripper_files[:seq_len]
    kept_indices = compute_trim_indices(
        camera_files=camera_files,
        trim_start_seconds=trim_start_seconds,
        trim_end_seconds=trim_end_seconds,
    )

    rgb_frames = []
    pose_series = []
    gripper_widths = []
    for idx in kept_indices:
        image_path = camera_dir / camera_files[idx]
        pose_path = pose_dir / pose_files[idx]
        gripper_path = gripper_dir / gripper_files[idx]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing RGB file: {image_path}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")
        if not gripper_path.is_file():
            raise FileNotFoundError(f"Missing gripper file: {gripper_path}")

        rgb_frames.append(read_rgb_image(image_path))
        pose_series.append(read_pose_json(pose_path))
        gripper_widths.append(read_gripper_width(gripper_path))

    pose_array = np.stack(pose_series, axis=0).astype(np.float32)
    rgb_array = np.stack(rgb_frames, axis=0).astype(np.uint8)
    gripper_array = np.asarray(gripper_widths, dtype=np.float32).reshape(-1, 1)

    trimmed_seq_len = pose_array.shape[0]
    start_pose = np.repeat(pose_array[:1], repeats=trimmed_seq_len, axis=0)
    end_pose = np.repeat(pose_array[-1:], repeats=trimmed_seq_len, axis=0)

    return {
        "camera0_rgb": rgb_array,
        "robot0_eef_pos": pose_array[:, :3],
        "robot0_eef_rot_axis_angle": pose_array[:, 3:],
        "robot0_gripper_width": gripper_array,
        "robot0_demo_start_pose": start_pose,
        "robot0_demo_end_pose": end_pose,
    }


def collect_episode_dirs(date_spec: DateSpec) -> List[Path]:
    if not date_spec.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {date_spec.input_root}")

    if date_spec.episode_list_path is not None and date_spec.episode_list_path.is_file():
        episode_names = read_episode_list(date_spec.episode_list_path)
        episode_dirs = [date_spec.input_root / name for name in episode_names]
        missing_episode_dirs = [path for path in episode_dirs if not path.is_dir()]
        if missing_episode_dirs:
            missing_preview = [str(path) for path in missing_episode_dirs[:10]]
            raise FileNotFoundError(
                f"Some episode directories do not exist under {date_spec.input_root}: "
                f"{missing_preview}"
            )
        return sorted(episode_dirs, key=episode_sort_key)

    episode_dirs = sorted(
        [
            path
            for path in date_spec.input_root.iterdir()
            if path.is_dir() and path.name.startswith("episode")
        ],
        key=episode_sort_key,
    )
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {date_spec.input_root}")
    return episode_dirs


def resolve_episode_list_path(input_root: Path) -> Path | None:
    episode_list_path = (
        REPO_ROOT
        / "umidata"
        / "data_process"
        / f"clean_report_{input_root.name}"
        / "structurally_usable_episodes.txt"
    )
    if episode_list_path.is_file():
        return episode_list_path
    return None


def resolve_pose_rel_path(input_root: Path, episode_list_path: Path | None) -> Path:
    candidate_episode_dirs: List[Path] = []
    if episode_list_path is not None:
        for episode_name in read_episode_list(episode_list_path):
            episode_dir = input_root / episode_name
            if episode_dir.is_dir():
                candidate_episode_dirs.append(episode_dir)
    else:
        candidate_episode_dirs = sorted(
            [
                path
                for path in input_root.iterdir()
                if path.is_dir() and path.name.startswith("episode")
            ],
            key=episode_sort_key,
        )

    for episode_dir in candidate_episode_dirs:
        for pose_rel_path in POSE_REL_PATH_CANDIDATES:
            if (episode_dir / pose_rel_path / "sync.txt").is_file():
                return pose_rel_path

    raise FileNotFoundError(
        f"Could not detect pose path under {input_root}. "
        "Expected one of: arm/endPose/sensorPose or arm/endPose/gripperPose"
    )


def resolve_date_specs(input_roots: List[Path]) -> List[DateSpec]:
    date_specs = []
    for input_root in input_roots:
        resolved_input_root = input_root.expanduser().resolve()
        if not resolved_input_root.is_dir():
            raise FileNotFoundError(f"Input root does not exist: {resolved_input_root}")
        episode_list_path = resolve_episode_list_path(resolved_input_root)
        pose_rel_path = resolve_pose_rel_path(
            input_root=resolved_input_root,
            episode_list_path=episode_list_path,
        )
        date_specs.append(
            DateSpec(
                date=resolved_input_root.name,
                input_root=resolved_input_root,
                pose_rel_path=pose_rel_path,
                episode_list_path=episode_list_path,
            )
        )
    return date_specs


def main() -> None:
    args = parse_args()
    if args.test_resize_image is not None:
        run_resize_image_test(
            input_image=args.test_resize_image,
            output_image=args.test_output_image,
        )
        return

    date_specs = resolve_date_specs(args.input_roots)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_episode_specs: List[Tuple[DateSpec, Path]] = []
    for date_spec in date_specs:
        episode_dirs = collect_episode_dirs(date_spec)
        print(
            f"Date {date_spec.date}: found {len(episode_dirs)} episodes "
            f"from {date_spec.input_root}"
        )
        if date_spec.episode_list_path is not None and date_spec.episode_list_path.is_file():
            print(f"  episode list: {date_spec.episode_list_path}")
        else:
            print("  episode list: not found, using all episode* directories")
        merged_episode_specs.extend((date_spec, episode_dir) for episode_dir in episode_dirs)

    if not merged_episode_specs:
        raise RuntimeError("No episodes were collected from the requested dates.")

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    success_count = 0
    skipped = []

    print(f"Output path: {output_path}")
    print(f"Trim start seconds: {args.trim_start_seconds}")
    print(f"Trim end seconds: {args.trim_end_seconds}")
    print(f"Total merged episodes: {len(merged_episode_specs)}")

    for date_spec, episode_dir in tqdm(merged_episode_specs, desc="Converting episodes"):
        merged_episode_name = f"{date_spec.date}/{episode_dir.name}"
        try:
            episode_data = build_episode_data(
                episode_dir=episode_dir,
                pose_rel_path=date_spec.pose_rel_path,
                trim_start_seconds=args.trim_start_seconds,
                trim_end_seconds=args.trim_end_seconds,
            )
            replay_buffer.add_episode(data=episode_data, compressors=None)
            success_count += 1
        except Exception as exc:
            skipped.append((merged_episode_name, str(exc)))
            print(f"[SKIP] {merged_episode_name}: {exc}")

    if success_count == 0:
        raise RuntimeError("No episodes were converted successfully.")

    if output_path.exists():
        output_path.unlink()

    with zarr.ZipStore(str(output_path), mode="w") as zip_store:
        replay_buffer.save_to_store(store=zip_store)

    print(f"Converted {success_count} episodes.")
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Saved merged dataset to: {output_path}")

    if skipped:
        print("Skipped episodes:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
