#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE_DIR = REPO_ROOT / "umidata" / "single" / "20260428"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "universal_manipulation_interface" / "openloop_validate" / "json_output"
START_INDEX = 10
END_INDEX = 20
OBS_HORIZON = 2
ACT_HORIZON = 16
DOWN_SAMPLE_STEPS = 3
CAMERA_NAME = "pikaGripperDepthCamera"  # or pikaGripperFisheyeCamera
POSE_REL_PATH = ""

@dataclass(frozen=True)
class AlignedFrame:
    timestamp: float
    image_b64: str
    pose7_xyzw: list[float]
    gripper_width: float


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Read RGB / pose / gripper data from one or more Pika episodes, align them by "
            "sync.txt, and export observation/action JSON pairs."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=DEFAULT_EPISODE_DIR,
        help=(
            "Single episode directory, e.g. pika_dataset/single/episode79, or an episode root "
            "directory that contains episode79, episode80, ..."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help="Start episode index (inclusive), e.g. 79 for episode79.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=END_INDEX,
        help="End episode index (inclusive), e.g. 82 for episode82.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=CAMERA_NAME,
        help="RGB camera directory name under camera/color/",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="pika",
        help="Pose device directory name under localization/pose/",
    )
    parser.add_argument(
        "--pose-rel-path",
        type=str,
        default=POSE_REL_PATH,
        help=(
            "Optional pose directory path relative to each episode, for example "
            "'arm/endPose/sensorPose'. When provided, it is tried before the built-in "
            "auto-detected pose locations."
        ),
    )
    parser.add_argument(
        "--gripper-device",
        type=str,
        default="pika",
        help="Gripper device directory name under gripper/encoder/",
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="arm_l",
        help="Arm key used in payload_raw.json.",
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=OBS_HORIZON,
        help="Number of aligned frames stored in each observation payload.",
    )
    parser.add_argument(
        "--act-horizon",
        type=int,
        default=ACT_HORIZON,
        help="Number of aligned future frames stored in each actions.json.",
    )
    parser.add_argument(
        "--down-sample-steps",
        type=int,
        default=DOWN_SAMPLE_STEPS,
        help=(
            "Temporal down-sampling step that matches the training sampler. "
            "For example, 3 means sampling every 3 raw frames."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Anchor stride in raw frames. Defaults to --down-sample-steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory. In multi-episode mode, each episode is exported to a child "
            "directory such as output-dir/episode79."
        ),
    )
    return parser.parse_args()


def read_sync_entries(sync_path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in sync_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"sync.txt is empty: {sync_path}")
    return entries


def encode_file_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("ascii")


def euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> list[float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]


def quat_xyzw_to_rotvec(quat_xyzw: Iterable[float]) -> list[float]:
    x, y, z, w = [float(value) for value in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    if w < 0.0:
        x = -x
        y = -y
        z = -z
        w = -w

    vec_norm = math.sqrt(x * x + y * y + z * z)
    if vec_norm < 1e-12:
        return [2.0 * x, 2.0 * y, 2.0 * z]

    angle = 2.0 * math.atan2(vec_norm, w)
    scale = angle / vec_norm
    return [x * scale, y * scale, z * scale]


def pose_json_to_pose7_xyzw(pose_json_path: Path) -> list[float]:
    raw_pose = json.loads(pose_json_path.read_text(encoding="utf-8"))
    position = [
        float(raw_pose["x"]),
        float(raw_pose["y"]),
        float(raw_pose["z"]),
    ]
    quat_xyzw = euler_xyz_to_quat_xyzw(
        roll=float(raw_pose["roll"]),
        pitch=float(raw_pose["pitch"]),
        yaw=float(raw_pose["yaw"]),
    )
    return position + quat_xyzw


def gripper_json_to_width(gripper_json_path: Path) -> float:
    raw_gripper = json.loads(gripper_json_path.read_text(encoding="utf-8"))
    return float(raw_gripper["distance"])


def pose7_xyzw_to_action(pose7_xyzw: Iterable[float], gripper_width: float) -> list[float]:
    pose7 = [float(x) for x in pose7_xyzw]
    if len(pose7) != 7:
        raise ValueError(f"Expected 7 values in pose7, got {len(pose7)}")
    position = pose7[:3]
    rotvec = quat_xyzw_to_rotvec(pose7[3:])
    return position + rotvec + [float(gripper_width)]


def is_single_episode_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_camera = (path / "camera").exists()
    has_pose = (path / "arm").exists() or (path / "localization").exists()
    has_gripper = (path / "gripper").exists()
    return has_camera and has_pose and has_gripper


def resolve_pose_and_gripper_dirs(
    episode_dir: Path,
    pose_device: str,
    gripper_device: str,
    pose_rel_path: str | None = None,
) -> tuple[Path, Path]:
    pose_candidates = []
    if pose_rel_path:
        pose_candidates.append(episode_dir / Path(pose_rel_path))
    pose_candidates.extend(
        [
            episode_dir / "arm" / "endPose" / "sensorPose",
            episode_dir / "arm" / "endPose" / "gripperPose",
            episode_dir / "localization" / "pose" / pose_device,
        ]
    )
    gripper_candidates = [
        episode_dir / "gripper" / "encoder" / "gripperWidth",
        episode_dir / "gripper" / "encoder" / gripper_device,
    ]

    pose_dir = next((path for path in pose_candidates if path.is_dir()), pose_candidates[0])
    gripper_dir = next(
        (path for path in gripper_candidates if path.is_dir()), gripper_candidates[0]
    )
    return pose_dir, gripper_dir


def parse_episode_index(episode_name: str) -> int:
    match = re.fullmatch(r"episode(\d+)", episode_name)
    if match is None:
        raise ValueError(f"Invalid episode directory name: {episode_name}")
    return int(match.group(1))


def resolve_episode_dirs(
    episode_dir: Path,
    start_index: int | None,
    end_index: int | None,
) -> list[tuple[int, Path]]:
    has_start = start_index is not None
    has_end = end_index is not None
    if has_start != has_end:
        raise ValueError("--start-index and --end-index must be provided together.")

    if has_start:
        assert start_index is not None
        assert end_index is not None
        if end_index < start_index:
            raise ValueError("--end-index must be greater than or equal to --start-index.")
        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode root directory does not exist: {episode_dir}")

        resolved_dirs: list[tuple[int, Path]] = []
        for episode_index in range(start_index, end_index + 1):
            this_episode_dir = episode_dir / f"episode{episode_index}"
            if not is_single_episode_dir(this_episode_dir):
                raise FileNotFoundError(
                    f"Episode directory is missing or incomplete: {this_episode_dir}"
                )
            resolved_dirs.append((episode_index, this_episode_dir))
        return resolved_dirs

    if not is_single_episode_dir(episode_dir):
        raise ValueError(
            "When --episode-dir points to an episode root directory, you must also provide "
            "--start-index and --end-index."
        )
    return [(parse_episode_index(episode_dir.name), episode_dir)]


def build_aligned_frames(
    episode_dir: Path,
    camera_name: str,
    pose_device: str,
    gripper_device: str,
    pose_rel_path: str | None = None,
) -> list[AlignedFrame]:
    rgb_dir = episode_dir / "camera" / "color" / camera_name
    pose_dir, gripper_dir = resolve_pose_and_gripper_dirs(
        episode_dir=episode_dir,
        pose_device=pose_device,
        gripper_device=gripper_device,
        pose_rel_path=pose_rel_path,
    )

    rgb_sync = read_sync_entries(rgb_dir / "sync.txt")
    pose_sync = read_sync_entries(pose_dir / "sync.txt")
    gripper_sync = read_sync_entries(gripper_dir / "sync.txt")

    aligned_count = min(len(rgb_sync), len(pose_sync), len(gripper_sync))
    if aligned_count == 0:
        raise ValueError("No aligned frames found from the three sync.txt files.")

    frames: list[AlignedFrame] = []
    for index in range(aligned_count):
        rgb_name = rgb_sync[index]
        pose_name = pose_sync[index]
        gripper_name = gripper_sync[index]

        rgb_path = rgb_dir / rgb_name
        pose_path = pose_dir / pose_name
        gripper_path = gripper_dir / gripper_name

        missing_paths = [path for path in (rgb_path, pose_path, gripper_path) if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing aligned files: {missing_text}")

        rgb_timestamp = float(Path(rgb_name).stem)
        pose_timestamp = float(Path(pose_name).stem)
        gripper_timestamp = float(Path(gripper_name).stem)
        aligned_timestamp = (rgb_timestamp + pose_timestamp + gripper_timestamp) / 3.0

        frames.append(
            AlignedFrame(
                timestamp=aligned_timestamp,
                image_b64=encode_file_base64(rgb_path),
                pose7_xyzw=pose_json_to_pose7_xyzw(pose_path),
                gripper_width=gripper_json_to_width(gripper_path),
            )
        )
    return frames


def build_observation_payload(
    arm_name: str,
    init_pose: list[float],
    obs_frames: list[AlignedFrame],
) -> dict:
    arm_payload = {
        "images": [frame.image_b64 for frame in obs_frames],
        "init_pose": init_pose,
        "poses": [frame.pose7_xyzw for frame in obs_frames],
        "grippers": [frame.gripper_width for frame in obs_frames],
        "timestamps": [frame.timestamp for frame in obs_frames],
    }
    return {
        arm_name: arm_payload,
        "send_timestamp": obs_frames[-1].timestamp,
        "type": "observation",
    }


def build_action_payload(action_frames: list[AlignedFrame]) -> dict:
    return {
        "actions": [
            pose7_xyzw_to_action(frame.pose7_xyzw, frame.gripper_width)
            for frame in action_frames
        ]
    }


def export_samples(
    aligned_frames: list[AlignedFrame],
    arm_name: str,
    obs_horizon: int,
    act_horizon: int,
    down_sample_steps: int,
    stride: int,
    output_dir: Path,
) -> int:
    if obs_horizon <= 0:
        raise ValueError("obs_horizon must be positive.")
    if act_horizon <= 0:
        raise ValueError("act_horizon must be positive.")
    if down_sample_steps <= 0:
        raise ValueError("down_sample_steps must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    min_required_frames = 1 + max(
        (obs_horizon - 1) * down_sample_steps,
        (act_horizon - 1) * down_sample_steps,
    )
    if len(aligned_frames) < min_required_frames:
        raise ValueError(
            "Aligned frames are not enough. Need at least "
            f"{min_required_frames}, got {len(aligned_frames)}."
        )

    init_pose = aligned_frames[0].pose7_xyzw
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    min_anchor_idx = (obs_horizon - 1) * down_sample_steps
    max_anchor_idx = len(aligned_frames) - 1 - (act_horizon - 1) * down_sample_steps
    for anchor_idx in range(min_anchor_idx, max_anchor_idx + 1, stride):
        obs_indices = list(
            range(
                anchor_idx - (obs_horizon - 1) * down_sample_steps,
                anchor_idx + 1,
                down_sample_steps,
            )
        )
        action_indices = list(
            range(
                anchor_idx,
                anchor_idx + act_horizon * down_sample_steps,
                down_sample_steps,
            )
        )
        obs_frames = [aligned_frames[idx] for idx in obs_indices]
        action_frames = [aligned_frames[idx] for idx in action_indices]
        sample_dir = output_dir / f"sample_{sample_count:06d}"
        sample_dir.mkdir(parents=True, exist_ok=False)

        observation_payload = build_observation_payload(
            arm_name=arm_name,
            init_pose=init_pose,
            obs_frames=obs_frames,
        )
        action_payload = build_action_payload(action_frames)

        (sample_dir / "payload_raw.json").write_text(
            json.dumps(observation_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "actions.json").write_text(
            json.dumps(action_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "meta.json").write_text(
            json.dumps(
                {
                    "anchor_index": anchor_idx,
                    "obs_indices": obs_indices,
                    "action_indices": action_indices,
                    "obs_horizon": obs_horizon,
                    "act_horizon": act_horizon,
                    "down_sample_steps": down_sample_steps,
                    "stride": stride,
                    "obs_timestamps": [frame.timestamp for frame in obs_frames],
                    "action_timestamps": [frame.timestamp for frame in action_frames],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        sample_count += 1

    return sample_count


def write_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    stride = args.down_sample_steps if args.stride is None else args.stride
    episode_dirs = resolve_episode_dirs(
        episode_dir=args.episode_dir,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    multi_episode_mode = len(episode_dirs) > 1 or args.start_index is not None

    batch_summaries: list[dict] = []
    for episode_index, episode_dir in episode_dirs:
        aligned_frames = build_aligned_frames(
            episode_dir=episode_dir,
            camera_name=args.camera_name,
            pose_device=args.pose_device,
            gripper_device=args.gripper_device,
            pose_rel_path=args.pose_rel_path,
        )
        episode_output_dir = args.output_dir / episode_dir.name if multi_episode_mode else args.output_dir
        sample_count = export_samples(
            aligned_frames=aligned_frames,
            arm_name=args.arm_name,
            obs_horizon=args.obs_horizon,
            act_horizon=args.act_horizon,
            down_sample_steps=args.down_sample_steps,
            stride=stride,
            output_dir=episode_output_dir,
        )

        summary = {
            "episode_index": episode_index,
            "episode_dir": str(episode_dir),
            "output_dir": str(episode_output_dir),
            "aligned_frame_count": len(aligned_frames),
            "obs_horizon": args.obs_horizon,
            "act_horizon": args.act_horizon,
            "down_sample_steps": args.down_sample_steps,
            "stride": stride,
            "sample_count": sample_count,
        }
        write_summary(summary, episode_output_dir / "export_summary.json")
        batch_summaries.append(summary)

    if multi_episode_mode:
        batch_summary = {
            "episode_root_dir": str(args.episode_dir),
            "output_root_dir": str(args.output_dir),
            "start_index": args.start_index,
            "end_index": args.end_index,
            "episode_count": len(batch_summaries),
            "episodes": batch_summaries,
        }
        write_summary(batch_summary, args.output_dir / "export_batch_summary.json")
        print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(batch_summaries[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
