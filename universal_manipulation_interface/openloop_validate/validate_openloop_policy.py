#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENLOOP_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = OPENLOOP_ROOT / "json_output"
DEFAULT_OUTPUT_DIR = OPENLOOP_ROOT / "validation_output"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realworld_deploy.policy_inference import PolicyInference


def load_realworld_defaults() -> dict[str, Any]:
    config_path = REPO_ROOT / "realworld_deploy" / "config.py"
    if not config_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("_openloop_validate_config", config_path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)

    return {
        "checkpoint_path": getattr(module, "POLICY_CHECKPOINT_PATH", None),
        "checkpoint_epoch": getattr(module, "EPOCH", "latest"),
        "preferred_arm": getattr(module, "DEFAULT_POLICY_ARM", "arm_l"),
        "crop": bool(getattr(module, "CROP", False)),
        "add_height": bool(getattr(module, "ADD_HEIGHT", False)),
        "height": float(getattr(module, "HEIGHT", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    defaults = load_realworld_defaults()

    parser = argparse.ArgumentParser(
        description=(
            "Run open-loop validation for exported episodes under json_output by reusing "
            "realworld_deploy/policy_inference.py."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory that contains episode folders such as episode149, episode150, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store per-episode validation JSON files and the batch summary.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(defaults["checkpoint_path"]) if defaults.get("checkpoint_path") else None,
        help="Checkpoint file or checkpoint directory accepted by PolicyInference.",
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=str,
        default=str(defaults.get("checkpoint_epoch", "latest")),
        help="Checkpoint epoch when --checkpoint-path points to a checkpoint directory.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=str(defaults.get("preferred_arm", "arm_l")),
        help="Preferred arm name when the payload format is ambiguous.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device used by PolicyInference, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        dest="episodes",
        default=[],
        help="Episode name to validate, e.g. episode149. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Only validate the first N samples in each episode.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when any sample fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from PolicyInference.",
    )
    parser.add_argument(
        "--save-policy-output",
        action="store_true",
        help="Enable PolicyInference internal JSON logging.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Enable PolicyInference image dumps for debugging.",
    )
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("crop", False)),
        help="Whether to crop images before resizing, matching the runtime deployment setting.",
    )
    parser.add_argument(
        "--add-height",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("add_height", False)),
        help="Whether to apply the deploy-time Z height offset to the sent action.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=float(defaults.get("height", 0.0)),
        help="Height offset in meters when --add-height is enabled.",
    )
    args = parser.parse_args()

    if args.checkpoint_path is None:
        parser.error(
            "Unable to infer a default checkpoint path from realworld_deploy/config.py. "
            "Please provide --checkpoint-path explicitly."
        )
    return args


def load_json(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def discover_episode_dirs(input_dir: Path, selected_episodes: list[str]) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if selected_episodes:
        episode_dirs = [input_dir / episode_name for episode_name in selected_episodes]
    else:
        episode_dirs = [path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("episode")]

    missing_dirs = [path for path in episode_dirs if not path.is_dir()]
    if missing_dirs:
        missing_text = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Episode directories do not exist: {missing_text}")

    return sorted(episode_dirs, key=lambda path: path.name)


def discover_sample_dirs(episode_dir: Path, sample_limit: int | None) -> list[Path]:
    sample_dirs = [path for path in episode_dir.iterdir() if path.is_dir() and path.name.startswith("sample_")]
    sample_dirs = sorted(sample_dirs, key=lambda path: path.name)
    if sample_limit is not None:
        sample_dirs = sample_dirs[:sample_limit]
    return sample_dirs


def detect_arm_name(payload: dict[str, Any], fallback_arm: str) -> str:
    available_arms = [arm_name for arm_name in ("arm_l", "arm_r") if isinstance(payload.get(arm_name), dict)]
    if len(available_arms) == 1:
        return available_arms[0]
    if fallback_arm in available_arms:
        return fallback_arm
    return fallback_arm


def summarize_errors(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def rotation_error_rad(pred_quat_xyzw: list[float], gt_rotvec: list[float]) -> float:
    pred_rot = Rotation.from_quat(np.asarray(pred_quat_xyzw, dtype=np.float64))
    gt_rot = Rotation.from_rotvec(np.asarray(gt_rotvec, dtype=np.float64))
    return float((pred_rot * gt_rot.inv()).magnitude())


def compare_action_chunks(
    predicted_actions: list[list[float]],
    gt_actions: list[list[float]],
    action_timestamps: list[float],
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    compared_steps = min(len(predicted_actions), len(gt_actions))
    per_step_results: list[dict[str, Any]] = []
    position_errors: list[float] = []
    rotation_errors_rad: list[float] = []

    for step_idx in range(compared_steps):
        pred_action = predicted_actions[step_idx]
        gt_action = gt_actions[step_idx]

        pred_position = np.asarray(pred_action[:3], dtype=np.float64)
        pred_quat_xyzw = pred_action[3:7]
        gt_position = np.asarray(gt_action[:3], dtype=np.float64)
        gt_rotvec = gt_action[3:6]

        pos_err = float(np.linalg.norm(pred_position - gt_position))
        rot_err_rad = rotation_error_rad(pred_quat_xyzw=pred_quat_xyzw, gt_rotvec=gt_rotvec)

        position_errors.append(pos_err)
        rotation_errors_rad.append(rot_err_rad)
        per_step_results.append(
            {
                "step_index": step_idx,
                "timestamp": action_timestamps[step_idx] if step_idx < len(action_timestamps) else None,
                "position_error_m": pos_err,
                "rotation_error_rad": rot_err_rad,
                "rotation_error_deg": float(math.degrees(rot_err_rad)),
            }
        )

    return per_step_results, position_errors, rotation_errors_rad


def validate_sample(
    policy: PolicyInference,
    sample_dir: Path,
    fallback_arm: str,
) -> tuple[dict[str, Any], list[float], list[float]]:
    payload_path = sample_dir / "payload_raw.json"
    actions_path = sample_dir / "actions.json"
    meta_path = sample_dir / "meta.json"

    payload = load_json(payload_path)
    gt_actions_payload = load_json(actions_path)
    meta = load_json(meta_path) if meta_path.exists() else {}

    arm_name = detect_arm_name(payload, fallback_arm=fallback_arm)
    # Compare against the raw logged action chunk so deploy-time height offsets do not skew metrics.
    log_response, _ = policy.infer(payload=payload, arm=arm_name)

    action_key = "action_l" if arm_name == "arm_l" else "action_r"
    predicted_actions = log_response.get(action_key, [])
    gt_actions = gt_actions_payload.get("actions", [])
    action_timestamps = meta.get("action_timestamps", [])

    per_step_results, position_errors, rotation_errors_rad = compare_action_chunks(
        predicted_actions=predicted_actions,
        gt_actions=gt_actions,
        action_timestamps=action_timestamps,
    )

    sample_result = {
        "sample_name": sample_dir.name,
        "arm_name": arm_name,
        "payload_path": str(payload_path),
        "actions_path": str(actions_path),
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "predicted_action_count": len(predicted_actions),
        "ground_truth_action_count": len(gt_actions),
        "compared_action_count": len(per_step_results),
        "position_error_m": summarize_errors(position_errors),
        "rotation_error_rad": summarize_errors(rotation_errors_rad),
        "rotation_error_deg": summarize_errors([math.degrees(value) for value in rotation_errors_rad]),
        "per_step": per_step_results,
    }
    return sample_result, position_errors, rotation_errors_rad


def validate_episode(
    policy: PolicyInference,
    episode_dir: Path,
    output_dir: Path,
    fallback_arm: str,
    sample_limit: int | None,
    fail_fast: bool,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    sample_dirs = discover_sample_dirs(episode_dir, sample_limit=sample_limit)
    if not sample_dirs:
        raise ValueError(f"No sample directories found under {episode_dir}")

    print(f"[openloop] validating {episode_dir.name} with {len(sample_dirs)} samples")
    policy.reset()

    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    all_position_errors: list[float] = []
    all_rotation_errors_rad: list[float] = []

    for sample_dir in sample_dirs:
        try:
            sample_result, position_errors, rotation_errors_rad = validate_sample(
                policy=policy,
                sample_dir=sample_dir,
                fallback_arm=fallback_arm,
            )
        except Exception as exc:
            failure = {
                "sample_name": sample_dir.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[openloop] failed {episode_dir.name}/{sample_dir.name}: {failure['error']}")
            if fail_fast:
                raise
            continue

        sample_results.append(sample_result)
        all_position_errors.extend(position_errors)
        all_rotation_errors_rad.extend(rotation_errors_rad)

    episode_result = {
        "episode_name": episode_dir.name,
        "episode_dir": str(episode_dir),
        "sample_count": len(sample_dirs),
        "successful_sample_count": len(sample_results),
        "failed_sample_count": len(failures),
        "position_error_m": summarize_errors(all_position_errors),
        "rotation_error_rad": summarize_errors(all_rotation_errors_rad),
        "rotation_error_deg": summarize_errors([math.degrees(value) for value in all_rotation_errors_rad]),
        "failures": failures,
        "run_config": run_config,
        "samples": sample_results,
    }

    write_json(output_dir / f"{episode_dir.name}_validation.json", episode_result)
    return episode_result


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_epoch": args.checkpoint_epoch,
        "arm": args.arm,
        "device": args.device,
        "crop": args.crop,
        "add_height": args.add_height,
        "height": args.height,
        "sample_limit": args.sample_limit,
        "episodes": args.episodes,
    }


def build_batch_summary(
    episode_results: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    all_position_errors: list[float] = []
    all_rotation_errors_rad: list[float] = []

    for episode_result in episode_results:
        for sample_result in episode_result.get("samples", []):
            for step_result in sample_result.get("per_step", []):
                all_position_errors.append(float(step_result["position_error_m"]))
                all_rotation_errors_rad.append(float(step_result["rotation_error_rad"]))

    return {
        "episode_count": len(episode_results),
        "successful_episode_count": sum(
            1 for episode_result in episode_results if episode_result.get("failed_sample_count", 0) == 0
        ),
        "position_error_m": summarize_errors(all_position_errors),
        "rotation_error_rad": summarize_errors(all_rotation_errors_rad),
        "rotation_error_deg": summarize_errors([math.degrees(value) for value in all_rotation_errors_rad]),
        "episodes": [
            {
                "episode_name": episode_result["episode_name"],
                "sample_count": episode_result["sample_count"],
                "successful_sample_count": episode_result["successful_sample_count"],
                "failed_sample_count": episode_result["failed_sample_count"],
                "position_error_m": episode_result["position_error_m"],
                "rotation_error_rad": episode_result["rotation_error_rad"],
                "rotation_error_deg": episode_result["rotation_error_deg"],
                "result_path": str(Path(run_config["output_dir"]) / f"{episode_result['episode_name']}_validation.json"),
            }
            for episode_result in episode_results
        ],
        "run_config": run_config,
    }


def main() -> None:
    args = parse_args()
    run_config = build_run_config(args)
    episode_dirs = discover_episode_dirs(args.input_dir, selected_episodes=args.episodes)

    policy = PolicyInference(
        checkpoint_path=str(args.checkpoint_path),
        checkpoint_epoch=args.checkpoint_epoch,
        preferred_arm=args.arm,
        device=args.device,
        verbose=args.verbose,
        _print=args.save_policy_output,
        img_save=args.save_images,
        crop=args.crop,
        add_height=args.add_height,
        height=args.height,
    )

    episode_results: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        episode_result = validate_episode(
            policy=policy,
            episode_dir=episode_dir,
            output_dir=args.output_dir,
            fallback_arm=args.arm,
            sample_limit=args.sample_limit,
            fail_fast=args.fail_fast,
            run_config=run_config,
        )
        episode_results.append(episode_result)

    batch_summary = build_batch_summary(episode_results=episode_results, run_config=run_config)
    summary_path = args.output_dir / "openloop_validation_summary.json"
    write_json(summary_path, batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"[openloop] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
