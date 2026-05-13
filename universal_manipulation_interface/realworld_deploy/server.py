#!/usr/bin/env python3
import socket
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

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
    VERBOSE,
    DATA_SAVE,
    PICT_SAVE,
    CROP,
    EPOCH,
    ACTION_CHUNK_HORIZON,
    ADD_HEIGHT,
    HEIGHT,
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

MAX_POSITION_STEP_M = 0.05
MAX_ROTATION_STEP_RAD = 0.1
MAX_GRIPPER_STEP_M = 0.08

MAX_OUTLIER_RATIO = 0.05

POLICY = PolicyInference(
    checkpoint_path=POLICY_CHECKPOINT_PATH,
    checkpoint_epoch=EPOCH,
    preferred_arm=DEFAULT_POLICY_ARM,
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


def reset_handler() -> None:
    global LATEST_OBSERVATION_PAYLOAD, LAST_ACTION_RESPONSE, LAST_LOG_ACTION_RESPONSE, LAST_INFERENCE_MONOTONIC, EPISODE_INIT_POSES
    POLICY.reset()
    LATEST_OBSERVATION_PAYLOAD = None
    LAST_ACTION_RESPONSE = None
    LAST_LOG_ACTION_RESPONSE = None
    LAST_INFERENCE_MONOTONIC = None
    EPISODE_INIT_POSES = {}


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


def _normalize_pose7(pose) -> Optional[list[float]]:
    if not isinstance(pose, (list, tuple)) or len(pose) != 7:
        return None
    try:
        return [float(value) for value in pose]
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
                clamped_step[7] = min(float(clamped_step[7]), MAX_GRIPPER_WIDTH_M)
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
    grip_step = np.abs(np.diff(action_array[:, 7], axis=0))
    try:
        rot_step = quaternion_step_angles(action_array[:, 3:7])
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
    arm = DEFAULT_POLICY_ARM
    for arm_name in ("arm_l", "arm_r"):
        if isinstance(payload.get(arm_name), dict):
            arm = arm_name
            break
    return {
        "type": "shakehands",
        "arm": arm,
        "sent_timestamp": time.time(),
        "received_timestamp": received_timestamp,
    }


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
                        ## shakehands
                        shakehands_response = prepare_observation(payload)
                        raw_sent_save_info = save_raw_sent_json(shakehands_response)
                        if VERBOSE:
                            print(
                                "[server] raw response saved | "
                                f"type={raw_sent_save_info['response_type']} "
                            )
                        # send_json_line(client_socket, shakehands_response, encoding=ENCODING)

                        ## run inference
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
