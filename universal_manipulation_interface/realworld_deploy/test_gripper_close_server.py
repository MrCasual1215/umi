#!/usr/bin/env python3
import socket
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from common import build_status_response, recv_json_line, save_response_record, send_json_line
from config import (
    ACTION_CHUNK_HORIZON,
    BUFFER_SIZE,
    ENCODING,
    MAX_CLIENTS,
    SERVER_HOST,
    SERVER_PORT,
    SOCKET_TIMEOUT_SEC,
    VERBOSE,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
RAW_SENT_JSON_ROOT = OUTPUT_ROOT / "raw_sent_json"
RAW_RECEIVED_JSON_ROOT = OUTPUT_ROOT / "raw_received_json"

DEFAULT_ARM = "arm_l"
DEFAULT_ACTION_HORIZON = ACTION_CHUNK_HORIZON if ACTION_CHUNK_HORIZON is not None else 16
MIN_GRIPPER_WIDTH_M = 0.03
MAX_GRIPPER_WIDTH_M = 0.10
GRIPPER_WIDTH_STEP_M = 0.005

CURRENT_GRIPPER_WIDTH_M = MIN_GRIPPER_WIDTH_M
GRIPPER_WIDTH_DIRECTION = 1.0


def _normalize_pose7(values: Optional[Sequence[float]]) -> Optional[list[float]]:
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _extract_reference_pose(observation_payload: Dict, arm_name: str) -> Optional[list[float]]:
    arm_payload = observation_payload.get(arm_name)
    if not isinstance(arm_payload, dict):
        return None

    init_pose = _normalize_pose7(arm_payload.get("init_pose"))
    if init_pose is not None:
        return init_pose

    arm_current_pose = _normalize_pose7(arm_payload.get("arm_current_pose"))
    if arm_current_pose is not None:
        return arm_current_pose

    poses = arm_payload.get("poses")
    if isinstance(poses, list) and poses:
        return _normalize_pose7(poses[0])

    return None


def _resolve_target_arm(observation_payload: Dict, preferred_arm: str = DEFAULT_ARM) -> str:
    for arm_name in (preferred_arm, "arm_l", "arm_r"):
        arm_payload = observation_payload.get(arm_name)
        if isinstance(arm_payload, dict) and _extract_reference_pose(observation_payload, arm_name) is not None:
            return arm_name
    raise ValueError("Missing valid reference pose for both arm_l and arm_r.")


def reset_gripper_wave() -> None:
    global CURRENT_GRIPPER_WIDTH_M, GRIPPER_WIDTH_DIRECTION
    CURRENT_GRIPPER_WIDTH_M = MIN_GRIPPER_WIDTH_M
    GRIPPER_WIDTH_DIRECTION = 1.0


def build_gripper_width_sequence(horizon: int) -> list[float]:
    global CURRENT_GRIPPER_WIDTH_M, GRIPPER_WIDTH_DIRECTION

    width_sequence = []
    for _ in range(max(1, int(horizon))):
        width_sequence.append(float(CURRENT_GRIPPER_WIDTH_M))

        next_width = CURRENT_GRIPPER_WIDTH_M + GRIPPER_WIDTH_DIRECTION * GRIPPER_WIDTH_STEP_M
        if next_width > MAX_GRIPPER_WIDTH_M:
            overflow = next_width - MAX_GRIPPER_WIDTH_M
            next_width = MAX_GRIPPER_WIDTH_M - overflow
            GRIPPER_WIDTH_DIRECTION = -1.0
        elif next_width < MIN_GRIPPER_WIDTH_M:
            overflow = MIN_GRIPPER_WIDTH_M - next_width
            next_width = MIN_GRIPPER_WIDTH_M + overflow
            GRIPPER_WIDTH_DIRECTION = 1.0

        CURRENT_GRIPPER_WIDTH_M = min(MAX_GRIPPER_WIDTH_M, max(MIN_GRIPPER_WIDTH_M, next_width))

    return width_sequence


def build_constant_close_action(
    observation_payload: Dict,
    arm_name: str = DEFAULT_ARM,
    horizon: int = DEFAULT_ACTION_HORIZON,
) -> Dict:
    arm_name = _resolve_target_arm(observation_payload, preferred_arm=arm_name)
    reference_pose = _extract_reference_pose(observation_payload, arm_name)
    if reference_pose is None:
        raise ValueError(f"Missing valid reference pose for {arm_name}.")

    gripper_widths = build_gripper_width_sequence(horizon)
    action_chunk = [list(reference_pose) + [gripper_width] for gripper_width in gripper_widths]

    response = {
        "type": "action",
        "action_l": action_chunk if arm_name == "arm_l" else [],
        "action_r": action_chunk if arm_name == "arm_r" else [],
        "timestamp": time.time(),
    }
    return response


def save_raw_sent_json(response: Dict) -> Dict:
    return save_response_record(response, RAW_SENT_JSON_ROOT)


def save_raw_received_json(payload: Dict) -> Dict:
    return save_response_record(payload, RAW_RECEIVED_JSON_ROOT)


def handle_message(payload: Dict) -> Dict:
    message_type = payload.get("type")
    if message_type == "reset":
        reset_gripper_wave()
        return build_status_response("reset_ack")
    if message_type != "observation":
        raise ValueError(f"Unsupported message type: {message_type}")

    response = build_constant_close_action(payload)
    return response


def serve_forever() -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(MAX_CLIENTS)

    print(
        f"[test_gripper_close_server] listening on {SERVER_HOST}:{SERVER_PORT} "
        f"| arm={DEFAULT_ARM} grip_range=[{MIN_GRIPPER_WIDTH_M:.4f}, {MAX_GRIPPER_WIDTH_M:.4f}] "
        f"step={GRIPPER_WIDTH_STEP_M:.4f} horizon={DEFAULT_ACTION_HORIZON}"
    )

    try:
        while True:
            print("[test_gripper_close_server] waiting for client connection...")
            client_socket, client_addr = server_socket.accept()
            print(f"[test_gripper_close_server] client connected: {client_addr}")
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
                    raw_received_info = save_raw_received_json(payload)
                    response = handle_message(payload)
                    save_info = save_raw_sent_json(response)

                    if VERBOSE:
                        print(
                            "[test_gripper_close_server] request received | "
                            f"type={raw_received_info['response_type']} path={raw_received_info['file_path']}"
                        )
                        print(
                            "[test_gripper_close_server] response prepared | "
                            f"type={response.get('type')} path={save_info['file_path']}"
                        )

                    send_json_line(client_socket, response, encoding=ENCODING)
            except ConnectionError:
                print(f"[test_gripper_close_server] client disconnected: {client_addr}")
            except socket.timeout:
                print(f"[test_gripper_close_server] client timeout: {client_addr}")
            except Exception as exc:
                print(f"[test_gripper_close_server] error while handling {client_addr}: {exc}")
            finally:
                client_socket.close()
    finally:
        server_socket.close()


if __name__ == "__main__":
    serve_forever()
