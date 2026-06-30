#!/usr/bin/env python3
import json
import socket
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from common import (
    build_status_response,
    recv_json_line,
    save_payload_record,
    save_response_record,
    send_json_line,
)
from policy_inference import PolicyInference
from config import (
    SERVER_HOST,
    SERVER_PORT,
    SOCKET_TIMEOUT_SEC,
    BUFFER_SIZE,
    ENCODING,
    MAX_CLIENTS,
    POLICY_CHECKPOINT_PATH,
    DEFAULT_POLICY_ARM,
    POLICY_ARM_MODE,
    VERBOSE,
    DATA_SAVE,
    PICT_SAVE,
    CROP,
    EPOCH,
    ACTION_CHUNK_HORIZON,
    ADD_HEIGHT,
    HEIGHT,
    ROBOT1_TO_ROBOT0_TX,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
SAVE_ROOT = OUTPUT_ROOT / "received_observations"
RAW_ACTION_ROOT = OUTPUT_ROOT / "raw_actions"
SENT_ROOT = OUTPUT_ROOT / "sent_actions"
RAW_RECEIVED_JSON_ROOT = OUTPUT_ROOT / "raw_received_json"
RAW_SENT_JSON_ROOT = OUTPUT_ROOT / "raw_sent_json"
EMPTY_SAVE_INFO = {"save_dir": None, "payload_type": None, "saved_images": []}
MAX_GRIPPER_WIDTH_M = 0.10
MAX_INFERENCE_RETRIES = 3

EPISODE_REPLAY_DIR = "/home/sunpeng/sp/umi_project/ldx/teledata/dual/episode_000008"
EPISODE_REPLAY_ENABLED = bool(EPISODE_REPLAY_DIR)

MAX_POSITION_STEP_M = 0.05
MAX_ROTATION_STEP_RAD = 0.1
MAX_GRIPPER_STEP_M = 0.08

MAX_OUTLIER_RATIO = 0.05

POLICY = None if EPISODE_REPLAY_ENABLED else PolicyInference(
    checkpoint_path=POLICY_CHECKPOINT_PATH,
    checkpoint_epoch=EPOCH,
    preferred_arm=DEFAULT_POLICY_ARM,
    arm_mode=POLICY_ARM_MODE,
    robot1_to_robot0_tx=ROBOT1_TO_ROBOT0_TX,
    verbose=VERBOSE,
    _print=DATA_SAVE,
    img_save=PICT_SAVE,
    crop=CROP,
    add_height=ADD_HEIGHT,
    height=HEIGHT,
)
LATEST_OBSERVATION_PAYLOAD = None
LAST_ACTION_RESPONSE = None
LAST_LOG_ACTION_RESPONSE = None
LAST_INFERENCE_MONOTONIC = None
EPISODE_INIT_POSES = {}
EPISODE_ACTION_CHUNK = None
EPISODE_ACTION_SOURCE = None


def reset_handler() -> None:
    global LATEST_OBSERVATION_PAYLOAD, LAST_ACTION_RESPONSE, LAST_LOG_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC, EPISODE_INIT_POSES, EPISODE_ACTION_CHUNK, EPISODE_ACTION_SOURCE
    if POLICY is not None:
        POLICY.reset()
    LATEST_OBSERVATION_PAYLOAD = None
    LAST_ACTION_RESPONSE = None
    LAST_LOG_ACTION_RESPONSE = None
    LAST_INFERENCE_MONOTONIC = None
    EPISODE_INIT_POSES = {}
    EPISODE_ACTION_CHUNK = None
    EPISODE_ACTION_SOURCE = None


def maybe_save_payload(payload: Dict) -> Dict:
    if not DATA_SAVE:
        return {
            **EMPTY_SAVE_INFO,
            "payload_type": payload.get("type"),
        }
    return save_payload_record(payload, SAVE_ROOT, PICT_SAVE)


def maybe_save_response(response: Dict) -> Dict:
    if not DATA_SAVE:
        return {
            "file_path": None,
            "response_type": response.get("type"),
        }
    return save_response_record(response, SENT_ROOT)


def maybe_save_raw_action(response: Dict) -> Dict:
    if not DATA_SAVE:
        return {
            "file_path": None,
            "response_type": response.get("type"),
        }
    return save_response_record(response, RAW_ACTION_ROOT)


def save_raw_received_json(payload: Dict) -> Dict:
    return save_response_record(payload, RAW_RECEIVED_JSON_ROOT)


def save_raw_sent_json(response: Dict) -> Dict:
    return save_response_record(response, RAW_SENT_JSON_ROOT)


def extract_observation_arm_payload(payload: Dict) -> Dict:
    if "arm_l" in payload or "arm_r" in payload:
        filtered_payload = {}
        for arm_name in ("arm_l", "arm_r"):
            arm_payload = payload.get(arm_name)
            if isinstance(arm_payload, dict):
                filtered_payload[arm_name] = arm_payload
        return filtered_payload
    return payload


def build_saved_inference_record(observation_payload: Dict, action_response: Dict) -> Dict:
    record = dict(action_response)
    record["observation"] = extract_observation_arm_payload(observation_payload)
    return record


def get_protocol_arm_names(payload: Optional[Dict] = None) -> list[str]:
    if POLICY_ARM_MODE == "bimanual":
        return ["arm_l", "arm_r"]

    return [
        select_response_arm(
            payload=payload,
            preferred_arm=DEFAULT_POLICY_ARM,
        )
    ]


def _normalize_pose7(pose) -> Optional[list[float]]:
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        return None
    try:
        return [float(value) for value in pose]
    except (TypeError, ValueError):
        return None


def _extract_replay_action_step(frame: Dict, arm_key: str) -> Optional[list[float]]:
    arm_frame = frame.get(arm_key)
    if not isinstance(arm_frame, dict):
        return None

    control = frame.get("control", {})
    if not isinstance(control, dict):
        return None
    control_arm = control.get(arm_key, {})
    if not isinstance(control_arm, dict):
        return None

    flange_pose = control_arm.get("feedback_fk_flange_pose", {})
    if not isinstance(flange_pose, dict):
        return None

    position = flange_pose.get("p")
    rotation_rpy = flange_pose.get("rpy")
    joint_feedback = arm_frame.get("q_fb")

    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return None
    if not isinstance(rotation_rpy, (list, tuple)) or len(rotation_rpy) != 3:
        return None
    if not isinstance(joint_feedback, (list, tuple)) or len(joint_feedback) < 1:
        return None

    try:
        quat_xyzw = Rotation.from_euler(
            "xyz",
            [
                float(rotation_rpy[0]),
                float(rotation_rpy[1]),
                float(rotation_rpy[2]),
            ],
            degrees=False,
        ).as_quat()
        return [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
            float(joint_feedback[-1]),
        ]
    except (TypeError, ValueError):
        return None


def _extract_init_pose_candidate(arm_payload: Dict) -> Optional[list[float]]:
    init_pose = _normalize_pose7(arm_payload.get("init_pose"))
    if init_pose is not None:
        return init_pose

    current_pose = _normalize_pose7(arm_payload.get("arm_current_pose"))
    if current_pose is not None:
        return current_pose

    poses = arm_payload.get("poses")
    if isinstance(poses, list) and poses:
        return _normalize_pose7(poses[0])

    return None


def read_sync_entries(sync_path: Path) -> list[str]:
    with sync_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def read_pose_step(pose_path: Path) -> list[float]:
    with pose_path.open("r", encoding="utf-8") as f:
        pose_dict = json.load(f)

    quat_xyzw = Rotation.from_euler(
        "xyz",
        [
            float(pose_dict["roll"]),
            float(pose_dict["pitch"]),
            float(pose_dict["yaw"]),
        ],
        degrees=False,
    ).as_quat()
    return [
        float(pose_dict["x"]),
        float(pose_dict["y"]),
        float(pose_dict["z"]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    ]


def read_gripper_width(gripper_path: Path) -> float:
    with gripper_path.open("r", encoding="utf-8") as f:
        gripper_dict = json.load(f)
    return float(gripper_dict["distance"])


def load_episode_action_chunk(episode_dir: Path) -> dict[str, list[list[float]]]:
    frames_path = episode_dir / "frames.jsonl"
    if frames_path.is_file():
        action_chunk = {"arm_l": [], "arm_r": []}
        with frames_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                frame = json.loads(stripped_line)
                left_step = _extract_replay_action_step(frame, "left")
                right_step = _extract_replay_action_step(frame, "right")

                if left_step is not None:
                    action_chunk["arm_l"].append(left_step)
                if right_step is not None:
                    action_chunk["arm_r"].append(right_step)

                if left_step is None and right_step is None:
                    raise ValueError(
                        "frames.jsonl line "
                        f"{line_idx} is missing replay fields for both arms."
                    )

        if not action_chunk["arm_l"] and not action_chunk["arm_r"]:
            raise ValueError(f"No valid replay actions found in {frames_path}")
        return action_chunk

    pose_dir = episode_dir / "arm" / "endPose" / "gripperPose"
    gripper_dir = episode_dir / "gripper" / "encoder" / "gripperWidth"

    pose_sync = pose_dir / "sync.txt"
    gripper_sync = gripper_dir / "sync.txt"
    missing_paths = [
        str(path) for path in (pose_sync, gripper_sync) if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Episode is missing required sync files: {missing_paths}"
        )

    pose_files = read_sync_entries(pose_sync)
    gripper_files = read_sync_entries(gripper_sync)
    if not pose_files or not gripper_files:
        raise ValueError(f"Episode sync.txt is empty: {episode_dir}")

    seq_len = min(len(pose_files), len(gripper_files))
    if seq_len <= 0:
        raise ValueError(f"Episode has no usable aligned steps: {episode_dir}")

    action_chunk = []
    for idx in range(seq_len):
        pose_path = pose_dir / pose_files[idx]
        gripper_path = gripper_dir / gripper_files[idx]
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")
        if not gripper_path.is_file():
            raise FileNotFoundError(f"Missing gripper file: {gripper_path}")

        action_chunk.append(read_pose_step(pose_path) + [read_gripper_width(gripper_path)])

    return {"arm_l": action_chunk, "arm_r": []}


def get_configured_episode_dir() -> Optional[Path]:
    if not EPISODE_REPLAY_DIR:
        return None

    episode_dir = Path(EPISODE_REPLAY_DIR).expanduser().resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(
            f"Configured episode dir does not exist: {episode_dir}"
        )
    return episode_dir


def ensure_episode_action_loaded() -> None:
    global EPISODE_ACTION_CHUNK, EPISODE_ACTION_SOURCE

    configured_episode_dir = get_configured_episode_dir()
    if configured_episode_dir is None:
        EPISODE_ACTION_CHUNK = None
        EPISODE_ACTION_SOURCE = None
        return

    episode_source = str(configured_episode_dir)
    if (
        EPISODE_ACTION_CHUNK is not None
        and EPISODE_ACTION_SOURCE == episode_source
    ):
        return

    EPISODE_ACTION_CHUNK = load_episode_action_chunk(configured_episode_dir)
    EPISODE_ACTION_SOURCE = episode_source

    if VERBOSE:
        loaded_arms = [
            arm_name for arm_name, steps in EPISODE_ACTION_CHUNK.items() if steps
        ]
        print(
            "[server] episode action loaded | "
            f"episode_dir={EPISODE_ACTION_SOURCE} "
            f"arms={loaded_arms} "
            f"steps_l={len(EPISODE_ACTION_CHUNK.get('arm_l', []))} "
            f"steps_r={len(EPISODE_ACTION_CHUNK.get('arm_r', []))}"
        )


def select_response_arm(
    payload: Optional[Dict],
    preferred_arm: Optional[str] = None,
) -> str:
    if preferred_arm in ("arm_l", "arm_r"):
        return preferred_arm
    if isinstance(payload, dict):
        available_arms = [
            arm_name
            for arm_name in ("arm_l", "arm_r")
            if isinstance(payload.get(arm_name), dict)
        ]
        if len(available_arms) == 1:
            return available_arms[0]
        if len(available_arms) > 1 and DEFAULT_POLICY_ARM in available_arms:
            return DEFAULT_POLICY_ARM
    return DEFAULT_POLICY_ARM


def populate_missing_init_pose(payload: Dict) -> Dict:
    global EPISODE_INIT_POSES

    for arm_name in ("arm_l", "arm_r"):
        arm_payload = payload.get(arm_name)
        if not isinstance(arm_payload, dict):
            continue

        if arm_payload.get("init_pose") is not None:
            normalized_init_pose = _normalize_pose7(arm_payload.get("init_pose"))
            if normalized_init_pose is not None:
                arm_payload["init_pose"] = normalized_init_pose
                EPISODE_INIT_POSES[arm_name] = normalized_init_pose
            continue

        cached_init_pose = EPISODE_INIT_POSES.get(arm_name)
        if cached_init_pose is not None:
            arm_payload["init_pose"] = list(cached_init_pose)
            continue

        init_pose_candidate = _extract_init_pose_candidate(arm_payload)
        if init_pose_candidate is not None:
            EPISODE_INIT_POSES[arm_name] = init_pose_candidate
            arm_payload["init_pose"] = list(init_pose_candidate)

    return payload


def truncate_action_chunk(response: Dict) -> Dict:
    horizon = ACTION_CHUNK_HORIZON
    if horizon is None:
        return response
    if horizon < 0:
        raise ValueError("ACTION_CHUNK_HORIZON must be greater than or equal to 0.")

    truncated_response = dict(response)
    for action_key in ("action_l", "action_r"):
        action_chunk = truncated_response.get(action_key)
        if isinstance(action_chunk, list):
            truncated_response[action_key] = action_chunk[:horizon]
    return truncated_response


def clamp_gripper_width(response: Dict) -> Dict:
    clamped_response = dict(response)
    for action_key in ("action_l", "action_r"):
        action_chunk = clamped_response.get(action_key)
        if not isinstance(action_chunk, list):
            continue

        clamped_chunk = []
        for step in action_chunk:
            if not isinstance(step, (list, tuple)) or len(step) < 8:
                clamped_chunk.append(step)
                continue

            clamped_step = list(step)
            try:
                clamped_step[-1] = min(float(clamped_step[-1]), MAX_GRIPPER_WIDTH_M)
            except (TypeError, ValueError):
                pass
            clamped_chunk.append(clamped_step)
        clamped_response[action_key] = clamped_chunk
    return clamped_response


def quaternion_step_angles(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Quaternion norm is too small.")
    normalized = quaternions / norms
    dots = np.sum(normalized[1:] * normalized[:-1], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def rotation_step_angles(action_array: np.ndarray) -> np.ndarray:
    if action_array.shape[1] >= 8:
        return quaternion_step_angles(action_array[:, 3:7])
    raise ValueError(f"unsupported_action_dim={action_array.shape[1]}")


def validate_action_chunk(action_chunk: list, action_key: str) -> tuple[bool, str]:
    if not action_chunk:
        return True, f"{action_key}: empty"
    if len(action_chunk) == 1:
        return True, f"{action_key}: single_step"

    try:
        action_array = np.asarray(action_chunk, dtype=np.float64)
    except (TypeError, ValueError):
        return False, f"{action_key}: non_numeric_action"

    if action_array.ndim != 2 or action_array.shape[1] < 8:
        return False, f"{action_key}: invalid_shape={action_array.shape}"
    if not np.isfinite(action_array).all():
        return False, f"{action_key}: non_finite_value"

    pos_step = np.linalg.norm(np.diff(action_array[:, :3], axis=0), axis=1)
    grip_step = np.abs(np.diff(action_array[:, -1], axis=0))
    try:
        rot_step = rotation_step_angles(action_array)
    except ValueError as exc:
        return False, f"{action_key}: {exc}"

    outlier_mask = (
        (pos_step > MAX_POSITION_STEP_M)
        | (rot_step > MAX_ROTATION_STEP_RAD)
        | (grip_step > MAX_GRIPPER_STEP_M)
    )
    outlier_ratio = float(np.mean(outlier_mask)) if outlier_mask.size else 0.0

    if outlier_ratio > MAX_OUTLIER_RATIO:
        return (
            False,
            f"{action_key}: outlier_ratio={outlier_ratio:.2f} "
            f"max_pos_step={float(np.max(pos_step)):.4f} "
            f"max_rot_step={float(np.max(rot_step)):.4f} "
            f"max_grip_step={float(np.max(grip_step)):.4f}",
        )

    return (
        True,
        f"{action_key}: smooth "
        f"outlier_ratio={outlier_ratio:.2f} "
        f"max_pos_step={float(np.max(pos_step)):.4f} "
        f"max_rot_step={float(np.max(rot_step)):.4f} "
        f"max_grip_step={float(np.max(grip_step)):.4f}",
    )


def validate_action_response(response: Dict) -> tuple[bool, str]:
    response_arm_mode = str(response.get("arm_mode", POLICY_ARM_MODE)).strip().lower()
    if response_arm_mode not in {"single", "bimanual"}:
        return False, f"invalid_arm_mode={response.get('arm_mode')!r}"

    if response_arm_mode == "bimanual":
        missing_keys = [
            action_key
            for action_key in ("action_l", "action_r")
            if not isinstance(response.get(action_key), list)
        ]
        if missing_keys:
            return False, f"bimanual response missing action keys: {missing_keys}"

    reasons = []
    for action_key in ("action_l", "action_r"):
        action_chunk = response.get(action_key)
        if not isinstance(action_chunk, list):
            continue
        is_valid, reason = validate_action_chunk(action_chunk, action_key)
        reasons.append(reason)
        if not is_valid:
            return False, "; ".join(reasons)
    if not reasons:
        return False, "no_action_chunk"
    return True, "; ".join(reasons)


def build_shakehands_response(payload: Dict, received_timestamp: float) -> Dict:
    arm_names = get_protocol_arm_names(payload)
    response = {
        "type": "shakehands",
        "arm_mode": POLICY_ARM_MODE,
        "arms": arm_names,
        "sent_timestamp": time.time(),
        "received_timestamp": received_timestamp,
    }
    if POLICY_ARM_MODE == "single":
        response["arm"] = arm_names[0]
    else:
        response["arm"] = None
    return response


def prepare_observation(payload: Dict) -> Dict:
    global LATEST_OBSERVATION_PAYLOAD, LAST_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC
    payload = populate_missing_init_pose(payload)
    LATEST_OBSERVATION_PAYLOAD = payload
    received_timestamp = time.time()
    arm_count = sum(
        1
        for arm_name in ("arm_l", "arm_r")
        if isinstance(payload.get(arm_name), dict)
    )

    if VERBOSE:
        message = (
            "[server] observation received | "
            f"payload_type={payload.get('type')}"
        )
        if arm_count:
            message += f" arms={arm_count}"
        if EPISODE_INIT_POSES:
            message += f" cached_init_pose_arms={sorted(EPISODE_INIT_POSES.keys())}"
        print(message)

    return build_shakehands_response(
        payload=LATEST_OBSERVATION_PAYLOAD,
        received_timestamp=received_timestamp,
    )


def run_observation_inference() -> tuple[Dict, Dict]:
    global LAST_ACTION_RESPONSE, LAST_LOG_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC
    if POLICY is None:
        raise RuntimeError("Policy inference is unavailable while episode replay mode is enabled.")
    maybe_save_payload(LATEST_OBSERVATION_PAYLOAD)

    total_inference_time = 0.0
    validation_reason = "unknown"
    selected_log_response = None
    selected_send_response = None

    for attempt_idx in range(1, MAX_INFERENCE_RETRIES + 1):
        start_time = time.monotonic()
        log_response, send_response = POLICY.infer(LATEST_OBSERVATION_PAYLOAD)
        end_time = time.monotonic()
        total_inference_time += end_time - start_time

        log_response = clamp_gripper_width(log_response)
        send_response = clamp_gripper_width(send_response)
        truncated_log_response = truncate_action_chunk(log_response)
        truncated_send_response = truncate_action_chunk(send_response)

        is_valid, validation_reason = validate_action_response(truncated_send_response)
        selected_log_response = truncated_log_response
        selected_send_response = truncated_send_response

        if VERBOSE:
            print(
                "[server] action chunk validation | "
                f"attempt={attempt_idx}/{MAX_INFERENCE_RETRIES} "
                f"valid={is_valid} "
                f"detail={validation_reason}"
            )

        if is_valid:
            break

    LAST_LOG_ACTION_RESPONSE = selected_log_response
    LAST_ACTION_RESPONSE = selected_send_response
    LAST_INFERENCE_MONOTONIC = time.monotonic()
    if VERBOSE:
        message = "[server] policy triggered immediately | "
        message += f"total_inference_time={total_inference_time:.6f}s "
        message += f"action_chunk_horizon={ACTION_CHUNK_HORIZON} "
        message += f"validation={validation_reason}"
        print(message)
    return LAST_LOG_ACTION_RESPONSE, LAST_ACTION_RESPONSE


def build_episode_action_response(observation_payload: Dict) -> Dict:
    global LAST_ACTION_RESPONSE, LAST_LOG_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC

    if EPISODE_ACTION_CHUNK is None:
        raise RuntimeError("No episode action chunk has been configured.")

    action_l = EPISODE_ACTION_CHUNK.get("arm_l", [])
    action_r = EPISODE_ACTION_CHUNK.get("arm_r", [])
    available_arms = [
        arm_name
        for arm_name, action_chunk in (("arm_l", action_l), ("arm_r", action_r))
        if action_chunk
    ]
    if not available_arms:
        raise RuntimeError("Configured episode does not contain any action chunk.")

    response_arm_mode = "bimanual" if len(available_arms) == 2 else "single"
    response_arm = (
        available_arms[0]
        if len(available_arms) == 1
        else select_response_arm(observation_payload)
    )
    response = {
        "type": "action",
        "arm_mode": response_arm_mode,
        "arms": available_arms,
        "arm": response_arm,
        "action_l": action_l,
        "action_r": action_r,
        "timestamp": time.time(),
    }
    response = clamp_gripper_width(response)
    LAST_LOG_ACTION_RESPONSE = response
    LAST_ACTION_RESPONSE = response
    LAST_INFERENCE_MONOTONIC = time.monotonic()

    if VERBOSE:
        print(
            "[server] episode action replay prepared | "
            f"source={EPISODE_ACTION_SOURCE} "
            f"arms={available_arms} "
            f"steps_l={len(action_l)} "
            f"steps_r={len(action_r)}"
        )
    return response


def handle_reset(_payload: Dict):
    reset_handler()
    if VERBOSE:
        print("[server] reset received")
    return build_status_response("reset_ack"), None, False


def handle_message(payload: Dict) -> tuple[Optional[Dict], Optional[Dict], bool]:
    message_type = payload.get("type")
    if VERBOSE:
        print(f"[server] received message type: {message_type}")
    if message_type == "reset":
        return handle_reset(payload)
    raise ValueError(f"Unsupported message type: {message_type}")

# TODO: 更新时间戳
def serve_forever() -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(MAX_CLIENTS)

    print(f"[server] listening on {SERVER_HOST}:{SERVER_PORT}")

    try:
        while True:
            print("[server] waiting for client connection...")
            client_socket, client_addr = server_socket.accept()
            print(f"[server] client connected: {client_addr}")
            client_socket.settimeout(SOCKET_TIMEOUT_SEC)
            recv_buffer = b""

            try:
                while True:
                    payload, recv_buffer = recv_json_line(
                        client_socket,
                        recv_buffer,
                        encoding=ENCODING,
                        buffer_size=BUFFER_SIZE,
                    )
                    raw_received_save_info = save_raw_received_json(payload)
                    if VERBOSE:
                        print(
                            "[server] raw request saved | "
                            f"type={raw_received_save_info['response_type']} "
                        )
                    if payload.get("type") == "observation":
                        ensure_episode_action_loaded()
                        ## shakehands
                        shakehands_response = prepare_observation(payload)
                        raw_sent_save_info = save_raw_sent_json(shakehands_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        # send_json_line(client_socket, shakehands_response, encoding=ENCODING)

                        ## run inference or replay configured episode action
                        if EPISODE_ACTION_CHUNK is not None:
                            log_response = build_episode_action_response(
                                LATEST_OBSERVATION_PAYLOAD
                            )
                            response = log_response
                        else:
                            log_response, response = run_observation_inference()
                        raw_action_save_info = maybe_save_raw_action(log_response)
                        if VERBOSE and raw_action_save_info["file_path"] is not None:
                            print(
                                "[server] raw action saved | "
                                f"type={raw_action_save_info['response_type']} "
                                # f"path={raw_action_save_info['file_path']}"
                            )
                        saved_response = build_saved_inference_record(
                            observation_payload=LATEST_OBSERVATION_PAYLOAD,
                            action_response=log_response,
                        )
                        response_save_info = maybe_save_response(saved_response)
                        if VERBOSE and response_save_info["file_path"] is not None:
                            print(
                                "[server] response saved | "
                                f"type={response_save_info['response_type']} "
                                # f"path={response_save_info['file_path']}"
                            )
                        raw_sent_save_info = save_raw_sent_json(log_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        send_json_line(client_socket, response, encoding=ENCODING)
                        if EPISODE_ACTION_CHUNK is not None:
                            if VERBOSE:
                                print(
                                    "[server] episode action replay finished, shutting down server"
                                )
                            return
                        continue

                    immediate_response, response, should_save_response = handle_message(payload)
                    if immediate_response is not None:
                        raw_sent_save_info = save_raw_sent_json(immediate_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        send_json_line(client_socket, immediate_response, encoding=ENCODING)
                    if response is None:
                        continue
                    if should_save_response:
                        raw_action_save_info = maybe_save_raw_action(response)
                        if VERBOSE and raw_action_save_info["file_path"] is not None:
                            print(
                                "[server] raw action saved | "
                                f"type={raw_action_save_info['response_type']} "
                                # f"path={raw_action_save_info['file_path']}"
                            )
                        saved_response = build_saved_inference_record(
                            observation_payload=LATEST_OBSERVATION_PAYLOAD,
                            action_response=response,
                        )
                        response_save_info = maybe_save_response(saved_response)
                        if VERBOSE and response_save_info["file_path"] is not None:
                            print(
                                "[server] response saved | "
                                f"type={response_save_info['response_type']} "
                                # f"path={response_save_info['file_path']}"
                            )
                    raw_sent_save_info = save_raw_sent_json(response)
                    if VERBOSE:
                        print(
                            "[server] raw response saved | "
                            f"type={raw_sent_save_info['response_type']} "
                        )
                    send_json_line(client_socket, response, encoding=ENCODING)
            except ConnectionError:
                print(f"[server] client disconnected: {client_addr}")
            except socket.timeout:
                print(f"[server] client timeout: {client_addr}")
            except Exception as exc:
                print(f"[server] error while handling {client_addr}: {exc}")
            finally:
                client_socket.close()
    finally:
        server_socket.close()


if __name__ == "__main__":
    serve_forever()
