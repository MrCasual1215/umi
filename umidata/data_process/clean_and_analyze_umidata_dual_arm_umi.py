#!/usr/bin/env python3
"""
Non-destructive data cleaning and quality analysis for dual-arm UMI episodes.

The script scans all `episode*` folders under one date directory, validates the
expected stream directories and payload files, extracts each episode's first
frame pose/time metadata, detects likely outliers, and writes reports and plots
under an output directory.

For each stream the loader prefers `sync.txt` when it contains usable entries.
If `sync.txt` is missing or empty, the script falls back to sorting payload
filenames by timestamp so partially synchronized recordings can still be
analyzed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "umidata" / "double" / "20260610_shorts"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "umidata" / "data_process" / "clean_report_20260610_shorts_dual"
)
LENGTH_MISMATCH_TOLERANCE = 5

LEFT_DEPTH_CAMERA_REL_PATH = Path("camera/color/pikaDepthCamera_l")
RIGHT_DEPTH_CAMERA_REL_PATH = Path("camera/color/pikaDepthCamera_r")
LEFT_FISHEYE_CAMERA_REL_PATH = Path("camera/color/pikaFisheyeCamera_l")
RIGHT_FISHEYE_CAMERA_REL_PATH = Path("camera/color/pikaFisheyeCamera_r")
LEFT_POSE_REL_PATH = Path("localization/pose/pika_l")
RIGHT_POSE_REL_PATH = Path("localization/pose/pika_r")
LEFT_GRIPPER_REL_PATH = Path("gripper/encoder/pika_l")
RIGHT_GRIPPER_REL_PATH = Path("gripper/encoder/pika_r")

STREAM_SPECS = {
    "left_depth_camera": {
        "dir": LEFT_DEPTH_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "right_depth_camera": {
        "dir": RIGHT_DEPTH_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "left_fisheye_camera": {
        "dir": LEFT_FISHEYE_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "right_fisheye_camera": {
        "dir": RIGHT_FISHEYE_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "left_pose": {
        "dir": LEFT_POSE_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
    "right_pose": {
        "dir": RIGHT_POSE_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
    "left_gripper": {
        "dir": LEFT_GRIPPER_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
    "right_gripper": {
        "dir": RIGHT_GRIPPER_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
}

DELTA_LABELS = {
    "left_depth_left_pose": "left_depth - left_pose",
    "right_depth_right_pose": "right_depth - right_pose",
    "left_fisheye_left_pose": "left_fisheye - left_pose",
    "right_fisheye_right_pose": "right_fisheye - right_pose",
    "left_gripper_left_pose": "left_gripper - left_pose",
    "right_gripper_right_pose": "right_gripper - right_pose",
    "right_depth_left_depth": "right_depth - left_depth",
    "right_pose_left_pose": "right_pose - left_pose",
    "right_gripper_left_gripper": "right_gripper - left_gripper",
}

CSV_FIELDNAMES = [
    "episode",
    "status",
    "error_count",
    "warning_count",
    "is_position_outlier",
    "is_alignment_outlier",
    "first_left_pose_timestamp",
    "first_right_pose_timestamp",
    "first_left_depth_timestamp",
    "first_right_depth_timestamp",
    "first_left_fisheye_timestamp",
    "first_right_fisheye_timestamp",
    "first_left_gripper_timestamp",
    "first_right_gripper_timestamp",
    "first_left_pose_x",
    "first_left_pose_y",
    "first_left_pose_z",
    "first_right_pose_x",
    "first_right_pose_y",
    "first_right_pose_z",
    "first_left_gripper_width",
    "first_right_gripper_width",
    "count_left_pose",
    "count_right_pose",
    "count_left_gripper",
    "count_right_gripper",
    "count_left_depth_camera",
    "count_right_depth_camera",
    "count_left_fisheye_camera",
    "count_right_fisheye_camera",
    "first_delta_left_depth_minus_left_pose",
    "first_delta_right_depth_minus_right_pose",
    "first_delta_left_fisheye_minus_left_pose",
    "first_delta_right_fisheye_minus_right_pose",
    "first_delta_left_gripper_minus_left_pose",
    "first_delta_right_gripper_minus_right_pose",
    "first_delta_right_depth_minus_left_depth",
    "first_delta_right_pose_minus_left_pose",
    "first_delta_right_gripper_minus_left_gripper",
    "max_abs_delta_left_depth_left_pose",
    "max_abs_delta_right_depth_right_pose",
    "max_abs_delta_left_fisheye_left_pose",
    "max_abs_delta_right_fisheye_right_pose",
    "max_abs_delta_left_gripper_left_pose",
    "max_abs_delta_right_gripper_right_pose",
    "max_abs_delta_right_depth_left_depth",
    "max_abs_delta_right_pose_left_pose",
    "max_abs_delta_right_gripper_left_gripper",
    "empty_or_missing_payload_files",
    "sync_fallback_streams",
    "length_mismatch_streams",
    "issues",
    "warnings",
]


@dataclass
class EpisodeResult:
    episode: str
    first_timestamps: Dict[str, float | None] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    durations: Dict[str, float | None] = field(default_factory=dict)
    first_pose_xyz: Dict[str, Tuple[float, float, float] | None] = field(
        default_factory=dict
    )
    first_gripper_widths: Dict[str, float | None] = field(default_factory=dict)
    max_abs_deltas: Dict[str, float | None] = field(default_factory=dict)
    first_deltas: Dict[str, float | None] = field(default_factory=dict)
    fallback_streams: List[str] = field(default_factory=list)
    empty_payload_count: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_length_mismatch: bool = False
    is_position_outlier: bool = False
    is_alignment_outlier: bool = False
    status: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and analyze dual-arm UMI raw episodes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Date directory containing episode* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reports, manifests and plots are written.",
    )
    parser.add_argument(
        "--alignment-threshold-seconds",
        type=float,
        default=0.05,
        help="Warn when the max absolute timestamp delta exceeds this threshold.",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=3.5,
        help="Robust z-score threshold for flagging first-pose outliers.",
    )
    return parser.parse_args()


def episode_sort_key(path: Path) -> Tuple[int, str]:
    suffix = path.name.replace("episode", "")
    if suffix.isdigit():
        return (int(suffix), path.name)
    return (10**9, path.name)


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def parse_timestamp_from_name(name: str) -> float:
    return float(Path(name).stem)


def parse_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def serialize_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def robust_outlier_mask(values: np.ndarray, z_threshold: float) -> np.ndarray:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.where(mad == 0.0, np.nan, mad)
    robust_z = 0.67448975 * (values - median) / scale
    robust_z = np.nan_to_num(robust_z, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return np.any(np.abs(robust_z) > z_threshold, axis=1)


def load_first_pose_xyz(path: Path) -> Tuple[float, float, float]:
    pose = parse_json(path)
    return (float(pose["x"]), float(pose["y"]), float(pose["z"]))


def load_first_gripper_width(path: Path) -> float:
    payload = parse_json(path)
    return float(payload["distance"])


def collect_sorted_payload_names(stream_dir: Path, suffix: str) -> List[str]:
    if not stream_dir.is_dir():
        return []
    try:
        files = sorted(
            [path for path in stream_dir.glob(f"*{suffix}") if path.is_file()],
            key=lambda path: parse_timestamp_from_name(path.name),
        )
    except Exception:
        files = sorted(
            [path for path in stream_dir.glob(f"*{suffix}") if path.is_file()],
            key=lambda path: path.name,
        )
    return [path.name for path in files]


def check_payload_files(base_dir: Path, names: Sequence[str]) -> Tuple[int, int, List[str]]:
    missing_or_empty = 0
    bad_names: List[str] = []
    for name in names:
        payload_path = base_dir / name
        if not payload_path.is_file():
            missing_or_empty += 1
            if len(bad_names) < 5:
                bad_names.append(f"missing:{name}")
            continue
        if payload_path.stat().st_size <= 0:
            missing_or_empty += 1
            if len(bad_names) < 5:
                bad_names.append(f"empty:{name}")
    return len(names), missing_or_empty, bad_names


def load_stream_names(
    episode_dir: Path,
    stream_name: str,
    result: EpisodeResult,
) -> List[str]:
    spec = STREAM_SPECS[stream_name]
    stream_dir = episode_dir / spec["dir"]
    sync_path = stream_dir / spec["sync_file"]

    if not stream_dir.is_dir():
        result.issues.append(f"missing stream directory: {stream_name}")
        return []

    names: List[str] = []
    used_fallback = False
    if sync_path.is_file() and sync_path.stat().st_size > 0:
        try:
            names = read_nonempty_lines(sync_path)
        except Exception as exc:
            result.issues.append(f"failed to read sync {stream_name}: {exc}")
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback or not names:
        fallback_names = collect_sorted_payload_names(stream_dir, spec["suffix"])
        if not fallback_names:
            if not used_fallback:
                result.issues.append(f"sync has no entries and no payload files: {stream_name}")
            else:
                result.issues.append(f"no payload files found: {stream_name}")
            return []
        if stream_name not in result.fallback_streams:
            result.fallback_streams.append(stream_name)
        names = fallback_names

    _, missing_or_empty, bad_names = check_payload_files(stream_dir, names)
    result.empty_payload_count += missing_or_empty
    if missing_or_empty > 0:
        result.issues.append(
            f"{stream_name} missing/empty payload count={missing_or_empty} samples={bad_names}"
        )
    return names


def safe_first_timestamp(names: Sequence[str]) -> float | None:
    if not names:
        return None
    return parse_timestamp_from_name(names[0])


def parse_timestamps(names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [parse_timestamp_from_name(name) for name in names],
        dtype=np.float64,
    )


def compute_stream_duration(timestamps: np.ndarray) -> float | None:
    if timestamps.size <= 0:
        return None
    return float(timestamps[-1] - timestamps[0])


def compute_pair_deltas(
    reference_names: Sequence[str], candidate_names: Sequence[str]
) -> Tuple[float | None, float | None]:
    if not reference_names or not candidate_names:
        return None, None

    reference_ts = parse_timestamps(reference_names)
    candidate_ts = parse_timestamps(candidate_names)
    insert_idx = np.searchsorted(candidate_ts, reference_ts, side="left")
    left_idx = np.clip(insert_idx - 1, 0, len(candidate_ts) - 1)
    right_idx = np.clip(insert_idx, 0, len(candidate_ts) - 1)

    left_delta = np.abs(candidate_ts[left_idx] - reference_ts)
    right_delta = np.abs(candidate_ts[right_idx] - reference_ts)
    use_right = right_delta < left_delta
    best_idx = np.where(use_right, right_idx, left_idx)
    deltas = reference_ts - candidate_ts[best_idx]
    return float(deltas[0]), float(np.max(np.abs(deltas)))


def analyze_episode(
    episode_dir: Path, alignment_threshold_seconds: float
) -> EpisodeResult:
    result = EpisodeResult(episode=episode_dir.name)
    stream_names: Dict[str, List[str]] = {}

    for stream_name in STREAM_SPECS:
        names = load_stream_names(episode_dir=episode_dir, stream_name=stream_name, result=result)
        stream_names[stream_name] = names
        result.counts[stream_name] = len(names)

        if not names:
            result.first_timestamps[stream_name] = None
            result.durations[stream_name] = None
            continue

        try:
            timestamps = parse_timestamps(names)
        except Exception as exc:
            result.first_timestamps[stream_name] = None
            result.durations[stream_name] = None
            result.issues.append(f"bad timestamp in {stream_name}: {exc}")
            continue

        result.first_timestamps[stream_name] = safe_first_timestamp(names)
        result.durations[stream_name] = compute_stream_duration(timestamps)
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
            result.issues.append(f"non-monotonic timestamps: {stream_name}")

    length_groups = {
        "camera": [
            "left_depth_camera",
            "right_depth_camera",
            "left_fisheye_camera",
            "right_fisheye_camera",
        ],
        "pose": ["left_pose", "right_pose"],
        "gripper": ["left_gripper", "right_gripper"],
    }
    for group_name, group_streams in length_groups.items():
        group_counts = {
            stream_name: result.counts.get(stream_name, 0)
            for stream_name in group_streams
            if result.counts.get(stream_name, 0) > 0
        }
        if group_counts and (max(group_counts.values()) - min(group_counts.values())) > LENGTH_MISMATCH_TOLERANCE:
            result.has_length_mismatch = True
            result.warnings.append(f"{group_name} stream length mismatch: {group_counts}")

    duration_values = {
        stream_name: duration
        for stream_name, duration in result.durations.items()
        if duration is not None and result.counts.get(stream_name, 0) > 0
    }
    if duration_values:
        duration_range = max(duration_values.values()) - min(duration_values.values())
        if duration_range > alignment_threshold_seconds:
            result.warnings.append(
                f"stream duration mismatch: range={duration_range:.6f}s values={duration_values}"
            )

    pose_streams = {
        "left_pose": LEFT_POSE_REL_PATH,
        "right_pose": RIGHT_POSE_REL_PATH,
    }
    for stream_name, rel_path in pose_streams.items():
        result.first_pose_xyz[stream_name] = None
        if not stream_names.get(stream_name):
            continue
        pose_path = episode_dir / rel_path / stream_names[stream_name][0]
        try:
            result.first_pose_xyz[stream_name] = load_first_pose_xyz(pose_path)
        except Exception as exc:
            result.issues.append(f"failed to parse first pose json {stream_name}: {exc}")

    gripper_streams = {
        "left_gripper": LEFT_GRIPPER_REL_PATH,
        "right_gripper": RIGHT_GRIPPER_REL_PATH,
    }
    for stream_name, rel_path in gripper_streams.items():
        result.first_gripper_widths[stream_name] = None
        if not stream_names.get(stream_name):
            continue
        gripper_path = episode_dir / rel_path / stream_names[stream_name][0]
        try:
            result.first_gripper_widths[stream_name] = load_first_gripper_width(gripper_path)
        except Exception as exc:
            result.issues.append(f"failed to parse first gripper json {stream_name}: {exc}")

    pair_definitions = {
        "left_depth_left_pose": ("left_depth_camera", "left_pose"),
        "right_depth_right_pose": ("right_depth_camera", "right_pose"),
        "left_fisheye_left_pose": ("left_fisheye_camera", "left_pose"),
        "right_fisheye_right_pose": ("right_fisheye_camera", "right_pose"),
        "left_gripper_left_pose": ("left_gripper", "left_pose"),
        "right_gripper_right_pose": ("right_gripper", "right_pose"),
        "right_depth_left_depth": ("right_depth_camera", "left_depth_camera"),
        "right_pose_left_pose": ("right_pose", "left_pose"),
        "right_gripper_left_gripper": ("right_gripper", "left_gripper"),
    }
    for metric_name, (left_stream, right_stream) in pair_definitions.items():
        try:
            first_delta, max_abs_delta = compute_pair_deltas(
                stream_names.get(left_stream, []),
                stream_names.get(right_stream, []),
            )
        except Exception as exc:
            first_delta, max_abs_delta = None, None
            result.issues.append(f"failed to compute delta {metric_name}: {exc}")

        result.first_deltas[metric_name] = first_delta
        result.max_abs_deltas[metric_name] = max_abs_delta
        if max_abs_delta is not None and max_abs_delta > alignment_threshold_seconds:
            result.warnings.append(
                f"alignment warning {metric_name}: max_abs_delta={max_abs_delta:.6f}s"
            )

    return result


def finalize_status(results: List[EpisodeResult], outlier_z_threshold: float) -> dict:
    pose_rows = []
    for idx, result in enumerate(results):
        left_xyz = result.first_pose_xyz.get("left_pose")
        right_xyz = result.first_pose_xyz.get("right_pose")
        if left_xyz is None or right_xyz is None or result.issues:
            continue
        pose_rows.append((idx, left_xyz + right_xyz))

    outlier_summary: dict = {
        "count": 0,
        "episodes": [],
        "median_xyzxyz": None,
        "mad_xyzxyz": None,
        "z_threshold": outlier_z_threshold,
    }
    if pose_rows:
        indices = np.asarray([idx for idx, _ in pose_rows], dtype=np.int64)
        points = np.asarray([xyzxyz for _, xyzxyz in pose_rows], dtype=np.float64)
        mask = robust_outlier_mask(points, outlier_z_threshold)
        median_xyzxyz = np.median(points, axis=0)
        mad_xyzxyz = np.median(np.abs(points - median_xyzxyz), axis=0)
        outlier_summary.update(
            {
                "count": int(np.count_nonzero(mask)),
                "episodes": [results[int(indices[i])].episode for i in np.where(mask)[0]],
                "median_xyzxyz": median_xyzxyz.tolist(),
                "mad_xyzxyz": mad_xyzxyz.tolist(),
            }
        )
        for local_idx, is_outlier in enumerate(mask):
            results[int(indices[local_idx])].is_position_outlier = bool(is_outlier)

    alignment_values = [
        value
        for result in results
        for value in result.max_abs_deltas.values()
        if value is not None and math.isfinite(value)
    ]
    alignment_threshold = float(np.percentile(alignment_values, 99)) if alignment_values else None

    for result in results:
        if alignment_threshold is not None:
            for value in result.max_abs_deltas.values():
                if value is not None and value > alignment_threshold:
                    result.is_alignment_outlier = True
                    break

        if result.issues:
            result.status = "invalid"
        elif result.is_position_outlier or result.is_alignment_outlier or result.warnings:
            result.status = "review"
        else:
            result.status = "valid"

    return {
        "position_outliers": outlier_summary,
        "alignment_outlier_threshold": alignment_threshold,
    }


def write_csv(results: Sequence[EpisodeResult], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
            left_xyz = result.first_pose_xyz.get("left_pose")
            right_xyz = result.first_pose_xyz.get("right_pose")
            writer.writerow(
                {
                    "episode": result.episode,
                    "status": result.status,
                    "error_count": len(result.issues),
                    "warning_count": len(result.warnings),
                    "is_position_outlier": int(result.is_position_outlier),
                    "is_alignment_outlier": int(result.is_alignment_outlier),
                    "first_left_pose_timestamp": serialize_float(result.first_timestamps.get("left_pose")),
                    "first_right_pose_timestamp": serialize_float(result.first_timestamps.get("right_pose")),
                    "first_left_depth_timestamp": serialize_float(
                        result.first_timestamps.get("left_depth_camera")
                    ),
                    "first_right_depth_timestamp": serialize_float(
                        result.first_timestamps.get("right_depth_camera")
                    ),
                    "first_left_fisheye_timestamp": serialize_float(
                        result.first_timestamps.get("left_fisheye_camera")
                    ),
                    "first_right_fisheye_timestamp": serialize_float(
                        result.first_timestamps.get("right_fisheye_camera")
                    ),
                    "first_left_gripper_timestamp": serialize_float(
                        result.first_timestamps.get("left_gripper")
                    ),
                    "first_right_gripper_timestamp": serialize_float(
                        result.first_timestamps.get("right_gripper")
                    ),
                    "first_left_pose_x": None if left_xyz is None else serialize_float(left_xyz[0]),
                    "first_left_pose_y": None if left_xyz is None else serialize_float(left_xyz[1]),
                    "first_left_pose_z": None if left_xyz is None else serialize_float(left_xyz[2]),
                    "first_right_pose_x": None if right_xyz is None else serialize_float(right_xyz[0]),
                    "first_right_pose_y": None if right_xyz is None else serialize_float(right_xyz[1]),
                    "first_right_pose_z": None if right_xyz is None else serialize_float(right_xyz[2]),
                    "first_left_gripper_width": serialize_float(
                        result.first_gripper_widths.get("left_gripper")
                    ),
                    "first_right_gripper_width": serialize_float(
                        result.first_gripper_widths.get("right_gripper")
                    ),
                    "count_left_pose": result.counts.get("left_pose", 0),
                    "count_right_pose": result.counts.get("right_pose", 0),
                    "count_left_gripper": result.counts.get("left_gripper", 0),
                    "count_right_gripper": result.counts.get("right_gripper", 0),
                    "count_left_depth_camera": result.counts.get("left_depth_camera", 0),
                    "count_right_depth_camera": result.counts.get("right_depth_camera", 0),
                    "count_left_fisheye_camera": result.counts.get("left_fisheye_camera", 0),
                    "count_right_fisheye_camera": result.counts.get("right_fisheye_camera", 0),
                    "first_delta_left_depth_minus_left_pose": serialize_float(
                        result.first_deltas.get("left_depth_left_pose")
                    ),
                    "first_delta_right_depth_minus_right_pose": serialize_float(
                        result.first_deltas.get("right_depth_right_pose")
                    ),
                    "first_delta_left_fisheye_minus_left_pose": serialize_float(
                        result.first_deltas.get("left_fisheye_left_pose")
                    ),
                    "first_delta_right_fisheye_minus_right_pose": serialize_float(
                        result.first_deltas.get("right_fisheye_right_pose")
                    ),
                    "first_delta_left_gripper_minus_left_pose": serialize_float(
                        result.first_deltas.get("left_gripper_left_pose")
                    ),
                    "first_delta_right_gripper_minus_right_pose": serialize_float(
                        result.first_deltas.get("right_gripper_right_pose")
                    ),
                    "first_delta_right_depth_minus_left_depth": serialize_float(
                        result.first_deltas.get("right_depth_left_depth")
                    ),
                    "first_delta_right_pose_minus_left_pose": serialize_float(
                        result.first_deltas.get("right_pose_left_pose")
                    ),
                    "first_delta_right_gripper_minus_left_gripper": serialize_float(
                        result.first_deltas.get("right_gripper_left_gripper")
                    ),
                    "max_abs_delta_left_depth_left_pose": serialize_float(
                        result.max_abs_deltas.get("left_depth_left_pose")
                    ),
                    "max_abs_delta_right_depth_right_pose": serialize_float(
                        result.max_abs_deltas.get("right_depth_right_pose")
                    ),
                    "max_abs_delta_left_fisheye_left_pose": serialize_float(
                        result.max_abs_deltas.get("left_fisheye_left_pose")
                    ),
                    "max_abs_delta_right_fisheye_right_pose": serialize_float(
                        result.max_abs_deltas.get("right_fisheye_right_pose")
                    ),
                    "max_abs_delta_left_gripper_left_pose": serialize_float(
                        result.max_abs_deltas.get("left_gripper_left_pose")
                    ),
                    "max_abs_delta_right_gripper_right_pose": serialize_float(
                        result.max_abs_deltas.get("right_gripper_right_pose")
                    ),
                    "max_abs_delta_right_depth_left_depth": serialize_float(
                        result.max_abs_deltas.get("right_depth_left_depth")
                    ),
                    "max_abs_delta_right_pose_left_pose": serialize_float(
                        result.max_abs_deltas.get("right_pose_left_pose")
                    ),
                    "max_abs_delta_right_gripper_left_gripper": serialize_float(
                        result.max_abs_deltas.get("right_gripper_left_gripper")
                    ),
                    "empty_or_missing_payload_files": result.empty_payload_count,
                    "sync_fallback_streams": "; ".join(sorted(result.fallback_streams)),
                    "length_mismatch_streams": int(result.has_length_mismatch),
                    "issues": " | ".join(result.issues),
                    "warnings": " | ".join(result.warnings),
                }
            )


def write_manifest(episodes: Iterable[str], path: Path) -> None:
    entries = list(episodes)
    with path.open("w", encoding="utf-8") as file:
        for episode in entries:
            file.write(f"{episode}\n")


def make_plot_env(output_dir: Path) -> None:
    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def save_plots(results: Sequence[EpisodeResult], output_dir: Path) -> List[str]:
    make_plot_env(output_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: failed to import matplotlib, skip plots: {exc}")
        return []

    plot_paths: List[str] = []

    pose_results = [
        result
        for result in results
        if result.first_pose_xyz.get("left_pose") is not None
        and result.first_pose_xyz.get("right_pose") is not None
    ]
    if pose_results:
        left_xyz = np.asarray(
            [result.first_pose_xyz["left_pose"] for result in pose_results],
            dtype=np.float64,
        )
        right_xyz = np.asarray(
            [result.first_pose_xyz["right_pose"] for result in pose_results],
            dtype=np.float64,
        )
        outlier_mask = np.asarray(
            [result.is_position_outlier for result in pose_results],
            dtype=bool,
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        labels = ["x", "y", "z"]
        for idx, axis in enumerate(axes[0]):
            axis.hist(left_xyz[:, idx], bins=24, alpha=0.85, color="#4472C4")
            axis.set_title(f"Left First Pose {labels[idx]}")
            axis.set_xlabel(labels[idx])
            axis.set_ylabel("Episode Count")
        for idx, axis in enumerate(axes[1]):
            axis.hist(right_xyz[:, idx], bins=24, alpha=0.85, color="#ED7D31")
            axis.set_title(f"Right First Pose {labels[idx]}")
            axis.set_xlabel(labels[idx])
            axis.set_ylabel("Episode Count")
        fig.tight_layout()
        plot_path = output_dir / "first_pose_histograms_dual.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, xyz, title in zip(
            axes,
            [left_xyz, right_xyz],
            ["Left First Pose XY", "Right First Pose XY"],
        ):
            axis.scatter(
                xyz[~outlier_mask, 0],
                xyz[~outlier_mask, 1],
                s=24,
                alpha=0.8,
                label="normal",
                color="#4CAF50",
            )
            if np.any(outlier_mask):
                axis.scatter(
                    xyz[outlier_mask, 0],
                    xyz[outlier_mask, 1],
                    s=40,
                    alpha=0.95,
                    label="position_outlier",
                    color="#E53935",
                )
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.legend()
        fig.tight_layout()
        plot_path = output_dir / "first_pose_xy_scatter_dual.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    relative_ts_rows = []
    for result in results:
        pose_ts = result.first_timestamps.get("left_pose")
        if pose_ts is not None:
            relative_ts_rows.append((result.episode, pose_ts))
    if relative_ts_rows:
        baseline = min(ts for _, ts in relative_ts_rows)
        episode_indices = [
            int(name.replace("episode", "")) if name.replace("episode", "").isdigit() else idx
            for idx, (name, _) in enumerate(relative_ts_rows)
        ]
        relative_ts = [ts - baseline for _, ts in relative_ts_rows]
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(episode_indices, relative_ts, marker="o", linestyle="-", markersize=3)
        axis.set_title("Left First Pose Time By Episode")
        axis.set_xlabel("Episode Index")
        axis.set_ylabel("Seconds Since Earliest Episode")
        fig.tight_layout()
        plot_path = output_dir / "left_first_pose_time_by_episode.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    metric_names = list(DELTA_LABELS.keys())
    metric_values = {
        name: [
            result.first_deltas.get(name)
            for result in results
            if result.first_deltas.get(name) is not None
        ]
        for name in metric_names
    }
    if any(metric_values[name] for name in metric_names):
        n_cols = 3
        n_rows = int(math.ceil(len(metric_names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes_array = np.asarray(axes).reshape(-1)
        for axis, name in zip(axes_array, metric_names):
            values = metric_values[name]
            if values:
                axis.hist(values, bins=24, alpha=0.85, color="#9C27B0")
            axis.set_title(f"First Delta {DELTA_LABELS[name]}")
            axis.set_xlabel("Seconds")
            axis.set_ylabel("Episode Count")
        for axis in axes_array[len(metric_names) :]:
            axis.axis("off")
        fig.tight_layout()
        plot_path = output_dir / "first_frame_alignment_histograms_dual.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    return plot_paths


def build_summary(
    input_root: Path,
    output_dir: Path,
    results: Sequence[EpisodeResult],
    plot_paths: Sequence[str],
    extra_summary: dict,
) -> dict:
    status_counter = Counter(result.status for result in results)
    issue_counter = Counter()
    for result in results:
        for issue in result.issues:
            issue_counter[issue.split(":")[0]] += 1

    valid_episodes = [result.episode for result in results if result.status == "valid"]
    review_episodes = [result.episode for result in results if result.status == "review"]
    invalid_episodes = [result.episode for result in results if result.status == "invalid"]

    return {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "episode_count": len(results),
        "status_counts": dict(status_counter),
        "valid_episode_count": len(valid_episodes),
        "review_episode_count": len(review_episodes),
        "invalid_episode_count": len(invalid_episodes),
        "valid_episodes": valid_episodes,
        "review_episodes": review_episodes,
        "invalid_episodes": invalid_episodes,
        "top_issue_categories": issue_counter.most_common(10),
        "plot_paths": list(plot_paths),
        **extra_summary,
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    episode_dirs = sorted(
        [path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("episode")],
        key=episode_sort_key,
    )
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        analyze_episode(
            episode_dir=episode_dir,
            alignment_threshold_seconds=args.alignment_threshold_seconds,
        )
        for episode_dir in episode_dirs
    ]
    extra_summary = finalize_status(
        results=results,
        outlier_z_threshold=args.outlier_z_threshold,
    )

    csv_path = output_dir / "episode_first_frame_report.csv"
    write_csv(results, csv_path)

    valid_episodes = [result.episode for result in results if result.status == "valid"]
    review_episodes = [result.episode for result in results if result.status == "review"]
    invalid_episodes = [result.episode for result in results if result.status == "invalid"]

    write_manifest(valid_episodes, output_dir / "valid_episodes.txt")
    write_manifest(review_episodes, output_dir / "review_episodes.txt")
    write_manifest(invalid_episodes, output_dir / "invalid_episodes.txt")
    write_manifest(
        [result.episode for result in results if result.status != "invalid"],
        output_dir / "structurally_usable_episodes.txt",
    )

    plot_paths = save_plots(results, output_dir)
    summary = build_summary(
        input_root=input_root,
        output_dir=output_dir,
        results=results,
        plot_paths=plot_paths,
        extra_summary=extra_summary,
    )
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=True, indent=2)

    print(f"Input root: {input_root}")
    print(f"Episodes scanned: {len(results)}")
    print(f"Valid episodes: {len(valid_episodes)}")
    print(f"Review episodes: {len(review_episodes)}")
    print(f"Invalid episodes: {len(invalid_episodes)}")
    print(f"CSV report: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    for plot_path in plot_paths:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
