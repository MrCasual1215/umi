# realworld_deploy Client API

## 1. Transport

- Protocol: TCP
- Host: read from `realworld_deploy/config.py`
- Port: read from `realworld_deploy/config.py`
- Encoding: `utf-8`
- Message framing: newline-delimited JSON
- One complete JSON message must end with `\n`

Example:

```text
{"type":"reset"}\n
{"type":"observation","arm_l":{...},"arm_r":{...}}\n
```

## 2. Current Deployment Mode

The current default deployment config is:

```python
TASK = "cloth"
MODE = "umi"
POLICY_MODEL = "transformer"
POLICY_ARM_MODE = "bimanual"
DEFAULT_POLICY_ARM = "arm_l"
```

Meaning:

- The server is currently configured for bimanual inference.
- In bimanual mode, the server expects both `arm_l` and `arm_r`.
- If later switched back to single-arm mode, the response still keeps both `action_l` and `action_r` fields, but only one side is non-empty.

## 3. Request Types

The server currently supports:

- `observation`
- `reset`

No other request type is accepted.

## 4. Observation Request

### 4.1 Recommended bimanual format

Use this format when `POLICY_ARM_MODE = "bimanual"`:

```json
{
  "type": "observation",
  "arm_l": {
    "images": ["<base64-jpeg>", "<base64-jpeg>"],
    "poses": [
      [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
      [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    ],
    "grippers": [0.02, 0.02],
    "init_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    "arm_current_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  },
  "arm_r": {
    "images": ["<base64-jpeg>", "<base64-jpeg>"],
    "poses": [
      [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
      [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
    ],
    "grippers": [0.03, 0.03],
    "init_pose": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
    "arm_current_pose": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
  }
}
```

### 4.2 Single-arm compatible format

When the server is switched to `POLICY_ARM_MODE = "single"`, there are two supported formats.

Recommended nested format:

```json
{
  "type": "observation",
  "arm_l": {
    "images": ["<base64-jpeg>", "<base64-jpeg>"],
    "poses": [
      [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
      [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    ],
    "grippers": [0.02, 0.02],
    "init_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  }
}
```

Legacy flat format:

```json
{
  "type": "observation",
  "images": ["<base64-jpeg>", "<base64-jpeg>"],
  "poses": [
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  ],
  "grippers": [0.02, 0.02],
  "init_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
}
```

## 5. Observation Field Definitions

### 5.1 `images`

- Type: `list[string]`
- Content: base64-encoded JPEG image
- Order: time history, old to new
- Current `cloth` config uses 1 RGB camera per arm and horizon 2
- Recommended count:
  - bimanual: each arm sends 2 frames
  - single-arm: active arm sends 2 frames

### 5.2 `poses`

- Type: `list[list[float]]`
- Each pose must be length 7:
  - `[x, y, z, qx, qy, qz, qw]`
- Quaternion order must be `xyzw`
- Order: time history, old to new
- Current `cloth` config uses low-dim horizon 2
- Recommended count:
  - bimanual: each arm sends 2 poses
  - single-arm: active arm sends 2 poses

### 5.3 `grippers`

- Type: `list[float]`
- Meaning: gripper width in meters
- Order: time history, old to new
- Current `cloth` config uses low-dim horizon 2
- Recommended count:
  - bimanual: each arm sends 2 values
  - single-arm: active arm sends 2 values

### 5.4 `init_pose`

- Type: `list[float]`, optional
- Length: 7
- Meaning: episode start pose for that arm

If omitted, the server will try the following fallback:

1. `arm_current_pose`
2. first item in `poses`

### 5.5 `arm_current_pose`

- Type: `list[float]`, optional
- Length: 7
- Used only as fallback when `init_pose` is absent

## 6. Observation Response

For an `observation` request, the client should expect exactly one socket response:

- `type = "action"`

Important:

- The server currently does not send the internal `shakehands` packet to the client.
- The server internally creates that object for logging only.
- So the client should not wait for a handshake response after sending `observation`.

### 6.1 Bimanual action response

```json
{
  "type": "action",
  "arm_mode": "bimanual",
  "arms": ["arm_l", "arm_r"],
  "action_l": [
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02],
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02]
  ],
  "action_r": [
    [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 0.03],
    [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 0.03]
  ],
  "timestamp": 1710000000.123
}
```

### 6.2 Single-arm action response

```json
{
  "type": "action",
  "arm_mode": "single",
  "arms": ["arm_l"],
  "action_l": [
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02],
    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02]
  ],
  "action_r": [],
  "timestamp": 1710000000.123
}
```

## 7. Action Field Definitions

### 7.1 `arm_mode`

- Type: `string`
- Value:
  - `single`
  - `bimanual`

### 7.2 `arms`

- Type: `list[string]`
- Meaning: which arms are active in this response
- Possible values:
  - `["arm_l"]`
  - `["arm_r"]`
  - `["arm_l", "arm_r"]`

### 7.3 `action_l` and `action_r`

- Type: `list[list[float]]`
- Each action step is length 8:
  - `[x, y, z, qx, qy, qz, qw, gripper_width]`
- Quaternion order is `xyzw`
- Position unit is meters
- Gripper unit is meters

Notes:

- `action_l` corresponds to left arm
- `action_r` corresponds to right arm
- In single-arm mode, one side is non-empty and the other side is `[]`
- In bimanual mode, both sides should normally be non-empty

### 7.4 Action horizon

- The server truncates the predicted chunk using `ACTION_CHUNK_HORIZON`
- Current value is `16`
- So the response normally contains at most 16 steps per arm

## 8. Reset Request

Request:

```json
{
  "type": "reset"
}
```

Response:

```json
{
  "type": "reset_ack",
  "timestamp": 1710000000.123
}
```

## 9. Error Handling

The server does not currently define a stable structured error JSON for inference failures.

If the payload is invalid, possible outcomes are:

- server logs an exception
- socket connection may be closed
- client may not receive a normal action response

Client recommendation:

1. set socket read timeout
2. if no valid action response is received, reconnect
3. send `reset` before restarting a new episode

## 10. Client Recommendations

### 10.1 Send order

For each control cycle:

1. collect observation history
2. send one `observation` JSON line
3. read one response JSON line
4. parse `type == "action"`
5. execute `action_l` and/or `action_r`

### 10.2 Bimanual recommendation

When in bimanual mode:

- always send both `arm_l` and `arm_r`
- keep both arms' history lengths aligned
- use the same timestamping policy on both arms on the client side

### 10.3 Image recommendation

- encode images as JPEG before base64
- keep image order consistent across time
- do not change camera assignment order during a running episode

## 11. Minimal Python Client Example

```python
import json
import socket


def send_json_line(sock, payload):
    sock.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))


def recv_json_line(sock):
    buffer = b""
    while b"\n" not in buffer:
        part = sock.recv(4096)
        if not part:
            raise ConnectionError("socket closed")
        buffer += part
    line, _ = buffer.split(b"\n", 1)
    return json.loads(line.decode("utf-8"))


payload = {
    "type": "observation",
    "arm_l": {
        "images": ["<base64-jpeg-0>", "<base64-jpeg-1>"],
        "poses": [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        ],
        "grippers": [0.02, 0.02]
    },
    "arm_r": {
        "images": ["<base64-jpeg-0>", "<base64-jpeg-1>"],
        "poses": [
            [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]
        ],
        "grippers": [0.03, 0.03]
    }
}


with socket.create_connection(("127.0.0.1", 8007), timeout=5.0) as sock:
    send_json_line(sock, payload)
    response = recv_json_line(sock)
    print(response["type"])
    print(response["arm_mode"])
    print(response["arms"])
```

## 12. Version Notes

This document matches the current server behavior:

- `observation -> action`
- `reset -> reset_ack`
- internal `shakehands` is not sent to the client
