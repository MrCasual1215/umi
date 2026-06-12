#!/usr/bin/env python3
"""
Non-destructive data cleaning and quality analysis for single-arm UMI episodes.

The script scans all `episode*` folders under one date directory, validates the
expected sync files and referenced payload files, extracts each episode's first
frame pose/time metadata, detects likely outliers, and writes reports and plots
under an output directory.
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
DEFAULT_INPUT_ROOT = REPO_ROOT / "umidata" / "single" / "20260603"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "umidata" / "data_process" / "clean_report_20260603"   

DEPTH_CAMERA_REL_PATH = Path("camera/color/pikaGripperDepthCamera")
FISHEYE_CAMERA_REL_PATH = Path("camera/color/pikaGripperFisheyeCamera")
POSE_REL_PATH = Path("arm/endPose/sensorPose")
GRIPPER_REL_PATH = Path("gripper/encoder/gripperWidth")

STREAM_SPECS = {
    "depth_camera": {
        "dir": DEPTH_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "fisheye_camera": {
        "dir": FISHEYE_CAMERA_REL_PATH,
        "suffix": ".jpg",
        "sync_file": "sync.txt",
    },
    "pose": {
        "dir": POSE_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
    "gripper": {
        "dir": GRIPPER_REL_PATH,
        "suffix": ".json",
        "sync_file": "sync.txt",
    },
}

CSV_FIELDNAMES = [
    "episode",
    "status",
    "error_count",
    "warning_count",
    "is_position_outlier",
    "is_alignment_outlier",
    "first_pose_timestamp",
    "first_depth_timestamp",
    "first_fisheye_timestamp",
    "first_gripper_timestamp",
    "first_pose_x",
    "first_pose_y",
    "first_pose_z",
    "first_gripper_width",
    "count_pose",
    "count_gripper",
    "count_depth_camera",
    "count_fisheye_camera",
    "first_delta_depth_minus_pose",
    "first_delta_fisheye_minus_pose",
    "first_delta_gripper_minus_pose",
    "max_abs_delta_depth_pose",
    "max_abs_delta_fisheye_pose",
    "max_abs_delta_gripper_pose",
    "empty_or_missing_payload_files",
    "missing_sync_streams",
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
    first_pose_xyz: Tuple[float, float, float] | None = None
    first_gripper_width: float | None = None
    max_abs_deltas: Dict[str, float | None] = field(default_factory=dict)
    first_deltas: Dict[str, float | None] = field(default_factory=dict)
    missing_sync_streams: List[str] = field(default_factory=list)
    empty_payload_count: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_length_mismatch: bool = False
    is_position_outlier: bool = False
    is_alignment_outlier: bool = False
    status: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and analyze single-arm UMI raw episodes."
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


def median_absolute_deviation(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


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


def safe_first_timestamp(names: Sequence[str]) -> float | None:
    if not names:
        return None
    return parse_timestamp_from_name(names[0])


def parse_timestamps(names: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_timestamp_from_name(name) for name in names], dtype=np.float64)


def compute_stream_duration(timestamps: np.ndarray) -> float | None:
    if timestamps.size <= 0:
        return None
    return float(timestamps[-1] - timestamps[0])


def compute_pair_deltas(
    left_names: Sequence[str], right_names: Sequence[str]
) -> Tuple[float | None, float | None]:
    pair_len = min(len(left_names), len(right_names))
    if pair_len <= 0:
        return None, None
    left_ts = parse_timestamps(left_names[:pair_len])
    right_ts = parse_timestamps(right_names[:pair_len])
    deltas = left_ts - right_ts
    return float(deltas[0]), float(np.max(np.abs(deltas)))


def analyze_episode(
    episode_dir: Path, alignment_threshold_seconds: float
) -> EpisodeResult:
    result = EpisodeResult(episode=episode_dir.name)
    sync_names: Dict[str, List[str]] = {}

    for stream_name, spec in STREAM_SPECS.items():
        stream_dir = episode_dir / spec["dir"]
        sync_path = stream_dir / spec["sync_file"]

        if not sync_path.is_file():
            result.missing_sync_streams.append(stream_name)
            result.issues.append(f"missing sync: {stream_name}")
            result.counts[stream_name] = 0
            result.first_timestamps[stream_name] = None
            sync_names[stream_name] = []
            continue

        if sync_path.stat().st_size <= 0:
            result.missing_sync_streams.append(stream_name)
            result.issues.append(f"empty sync: {stream_name}")
            result.counts[stream_name] = 0
            result.first_timestamps[stream_name] = None
            sync_names[stream_name] = []
            continue

        try:
            names = read_nonempty_lines(sync_path)
        except Exception as exc:
            result.issues.append(f"failed to read sync {stream_name}: {exc}")
            result.counts[stream_name] = 0
            result.first_timestamps[stream_name] = None
            sync_names[stream_name] = []
            continue

        if not names:
            result.issues.append(f"sync has no entries: {stream_name}")

        sync_names[stream_name] = names
        result.counts[stream_name] = len(names)
        result.durations[stream_name] = None

        try:
            timestamps = parse_timestamps(names)
        except Exception as exc:
            result.first_timestamps[stream_name] = None
            result.durations[stream_name] = None
            result.issues.append(f"bad timestamp in {stream_name}: {exc}")
            timestamps = np.asarray([], dtype=np.float64)
        else:
            result.first_timestamps[stream_name] = safe_first_timestamp(names)
            result.durations[stream_name] = compute_stream_duration(timestamps)
            if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
                result.issues.append(f"non-monotonic sync timestamps: {stream_name}")

        _, missing_or_empty, bad_names = check_payload_files(stream_dir, names)
        result.empty_payload_count += missing_or_empty
        if missing_or_empty > 0:
            result.issues.append(
                f"{stream_name} missing/empty payload count={missing_or_empty} samples={bad_names}"
            )

    count_values = [count for count in result.counts.values() if count > 0]
    if count_values and len(set(count_values)) != 1:
        result.has_length_mismatch = True
        result.warnings.append(f"sync length mismatch: {result.counts}")

    duration_values = {
        stream_name: duration
        for stream_name, duration in result.durations.items()
        if duration is not None and result.counts.get(stream_name, 0) > 0
    }
    if duration_values:
        duration_range = max(duration_values.values()) - min(duration_values.values())
        if duration_range > alignment_threshold_seconds:
            result.warnings.append(
                f"sync duration mismatch: range={duration_range:.6f}s values={duration_values}"
            )

    if sync_names.get("pose"):
        pose_path = episode_dir / POSE_REL_PATH / sync_names["pose"][0]
        try:
            result.first_pose_xyz = load_first_pose_xyz(pose_path)
        except Exception as exc:
            result.issues.append(f"failed to parse first pose json: {exc}")

    if sync_names.get("gripper"):
        gripper_path = episode_dir / GRIPPER_REL_PATH / sync_names["gripper"][0]
        try:
            result.first_gripper_width = load_first_gripper_width(gripper_path)
        except Exception as exc:
            result.issues.append(f"failed to parse first gripper json: {exc}")

    pair_definitions = {
        "depth_pose": ("depth_camera", "pose"),
        "fisheye_pose": ("fisheye_camera", "pose"),
        "gripper_pose": ("gripper", "pose"),
    }
    for metric_name, (left_stream, right_stream) in pair_definitions.items():
        try:
            first_delta, max_abs_delta = compute_pair_deltas(
                sync_names.get(left_stream, []), sync_names.get(right_stream, [])
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
    pose_rows = [
        (idx, result.first_pose_xyz)
        for idx, result in enumerate(results)
        if result.first_pose_xyz is not None and not result.issues
    ]
    outlier_summary: dict = {
        "count": 0,
        "episodes": [],
        "median_xyz": None,
        "mad_xyz": None,
        "z_threshold": outlier_z_threshold,
    }
    if pose_rows:
        indices = np.asarray([idx for idx, _ in pose_rows], dtype=np.int64)
        points = np.asarray([xyz for _, xyz in pose_rows], dtype=np.float64)
        mask = robust_outlier_mask(points, outlier_z_threshold)
        median_xyz = np.median(points, axis=0)
        mad_xyz = np.median(np.abs(points - median_xyz), axis=0)
        outlier_summary.update(
            {
                "count": int(np.count_nonzero(mask)),
                "episodes": [results[int(indices[i])].episode for i in np.where(mask)[0]],
                "median_xyz": median_xyz.tolist(),
                "mad_xyz": mad_xyz.tolist(),
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
            writer.writerow(
                {
                    "episode": result.episode,
                    "status": result.status,
                    "error_count": len(result.issues),
                    "warning_count": len(result.warnings),
                    "is_position_outlier": int(result.is_position_outlier),
                    "is_alignment_outlier": int(result.is_alignment_outlier),
                    "first_pose_timestamp": serialize_float(
                        result.first_timestamps.get("pose")
                    ),
                    "first_depth_timestamp": serialize_float(
                        result.first_timestamps.get("depth_camera")
                    ),
                    "first_fisheye_timestamp": serialize_float(
                        result.first_timestamps.get("fisheye_camera")
                    ),
                    "first_gripper_timestamp": serialize_float(
                        result.first_timestamps.get("gripper")
                    ),
                    "first_pose_x": None
                    if result.first_pose_xyz is None
                    else serialize_float(result.first_pose_xyz[0]),
                    "first_pose_y": None
                    if result.first_pose_xyz is None
                    else serialize_float(result.first_pose_xyz[1]),
                    "first_pose_z": None
                    if result.first_pose_xyz is None
                    else serialize_float(result.first_pose_xyz[2]),
                    "first_gripper_width": serialize_float(result.first_gripper_width),
                    "count_pose": result.counts.get("pose", 0),
                    "count_gripper": result.counts.get("gripper", 0),
                    "count_depth_camera": result.counts.get("depth_camera", 0),
                    "count_fisheye_camera": result.counts.get("fisheye_camera", 0),
                    "first_delta_depth_minus_pose": serialize_float(
                        result.first_deltas.get("depth_pose")
                    ),
                    "first_delta_fisheye_minus_pose": serialize_float(
                        result.first_deltas.get("fisheye_pose")
                    ),
                    "first_delta_gripper_minus_pose": serialize_float(
                        result.first_deltas.get("gripper_pose")
                    ),
                    "max_abs_delta_depth_pose": serialize_float(
                        result.max_abs_deltas.get("depth_pose")
                    ),
                    "max_abs_delta_fisheye_pose": serialize_float(
                        result.max_abs_deltas.get("fisheye_pose")
                    ),
                    "max_abs_delta_gripper_pose": serialize_float(
                        result.max_abs_deltas.get("gripper_pose")
                    ),
                    "empty_or_missing_payload_files": result.empty_payload_count,
                    "missing_sync_streams": "; ".join(result.missing_sync_streams),
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: List[str] = []

    pose_results = [result for result in results if result.first_pose_xyz is not None]
    if pose_results:
        pose_xyz = np.asarray([result.first_pose_xyz for result in pose_results], dtype=np.float64)
        outlier_mask = np.asarray(
            [result.is_position_outlier for result in pose_results], dtype=bool
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels = ["x", "y", "z"]
        for idx, axis in enumerate(axes):
            axis.hist(pose_xyz[:, idx], bins=24, alpha=0.85, color="#4472C4")
            axis.set_title(f"First Pose {labels[idx]} Distribution")
            axis.set_xlabel(labels[idx])
            axis.set_ylabel("Episode Count")
        fig.tight_layout()
        plot_path = output_dir / "first_pose_histograms.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

        fig, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            pose_xyz[~outlier_mask, 0],
            pose_xyz[~outlier_mask, 1],
            s=24,
            alpha=0.8,
            label="normal",
            color="#4CAF50",
        )
        if np.any(outlier_mask):
            axis.scatter(
                pose_xyz[outlier_mask, 0],
                pose_xyz[outlier_mask, 1],
                s=40,
                alpha=0.95,
                label="position_outlier",
                color="#E53935",
            )
        axis.set_title("First Pose XY Scatter")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend()
        fig.tight_layout()
        plot_path = output_dir / "first_pose_xy_scatter.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    relative_ts_rows = []
    for result in results:
        pose_ts = result.first_timestamps.get("pose")
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
        axis.set_title("First Pose Time By Episode")
        axis.set_xlabel("Episode Index")
        axis.set_ylabel("Seconds Since Earliest Episode")
        fig.tight_layout()
        plot_path = output_dir / "first_pose_time_by_episode.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(plot_path))

    delta_names = ["depth_pose", "fisheye_pose", "gripper_pose"]
    delta_labels = {
        "depth_pose": "depth - pose",
        "fisheye_pose": "fisheye - pose",
        "gripper_pose": "gripper - pose",
    }
    delta_values = {
        name: [
            result.first_deltas.get(name)
            for result in results
            if result.first_deltas.get(name) is not None
        ]
        for name in delta_names
    }
    if any(delta_values[name] for name in delta_names):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, name in zip(axes, delta_names):
            values = delta_values[name]
            if values:
                axis.hist(values, bins=24, alpha=0.85, color="#9C27B0")
            axis.set_title(f"First Delta {delta_labels[name]}")
            axis.set_xlabel("Seconds")
            axis.set_ylabel("Episode Count")
        fig.tight_layout()
        plot_path = output_dir / "first_frame_alignment_histograms.png"
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

    summary = {
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
    return summary


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
