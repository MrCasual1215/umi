import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_umi_action,
    get_real_umi_obs_dict,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class PolicyInference:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_epoch: Union[int, str] = "latest",
        preferred_arm: str = "arm_l",
        arm_mode: str = "single",
        robot1_to_robot0_tx: Optional[Sequence[Sequence[float]]] = None,
        device: Optional[str] = None,
        verbose: bool = True,
        _print: bool = True,
        img_save: bool = False,
        crop: bool = False,
        add_height: bool = False,
        height: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_epoch = checkpoint_epoch
        self.preferred_arm = preferred_arm
        self.arm_mode = self._normalize_arm_mode(arm_mode)
        self.robot1_to_robot0_tx = self._normalize_robot_transform(robot1_to_robot0_tx)
        self.verbose = verbose
        self.img_save = img_save
        self.print = _print
        self.crop = crop
        self.add_height = add_height
        self.height = float(height)

        self.ckpt_payload, self.cfg = self._load_checkpoint(self.checkpoint_path)
        self.shape_meta = self.cfg.task.shape_meta
        self.model_robot_count = self._infer_robot_count(self.shape_meta)
        self.rgb_keys = self._sorted_obs_keys(self.shape_meta["obs"], obs_type="rgb")
        self._validate_arm_mode_against_checkpoint()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = self._create_policy(self.cfg, self.ckpt_payload, self.device)
        self.obs_pose_repr = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr

        if self.verbose:
            print(
                "[policy] initialized | "
                f"device={self.device} "
                f"arm_mode={self.arm_mode} "
                f"model_robot_count={self.model_robot_count} "
                f"obs_pose_repr={self.obs_pose_repr} "
                f"action_pose_repr={self.action_pose_repr} "
                f"add_height={self.add_height} "
                f"height={self.height}"
            )

    def reset(self) -> None:
        self.policy.reset()
        if self.verbose:
            print("[policy] state reset")

    def infer(self, payload: Dict, arm: Optional[str] = None) -> Tuple[Dict, Dict]:
        arm_payload_items = self._resolve_arm_payloads(payload, requested_arm=arm)
        env_obs, episode_start_pose = self._build_env_obs(
            arm_payload_items=arm_payload_items,
            shape_meta=self.shape_meta,
            img_save=self.img_save,
        )
        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.shape_meta,
            obs_pose_repr=self.obs_pose_repr,
            tx_robot1_robot0=self.robot1_to_robot0_tx,
            episode_start_pose=(
                episode_start_pose if self._uses_episode_start_pose(self.shape_meta) else None
            ),
        )
        obs_dict = dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            start_time = time.time()
            result = self.policy.predict_action(obs_dict)
            end_time = time.time()

        raw_action = result["action_pred"][0].detach().cpu().numpy()
        env_action = get_real_umi_action(raw_action, env_obs, self.action_pose_repr)
        action_by_arm = self._split_env_action_by_arm(
            env_action=env_action,
            arm_names=[arm_name for arm_name, _ in arm_payload_items],
        )
        sent_action_by_arm = {
            arm_name: self._apply_height_offset(actions)
            for arm_name, actions in action_by_arm.items()
        }

        if self.print:
            self._save_policy_output(
                raw_action=raw_action,
                env_action=env_action,
                action_by_arm=action_by_arm,
            )

        log_response = self._build_action_response(action_by_arm)
        send_response = self._build_action_response(
            sent_action_by_arm,
            timestamp=log_response["timestamp"],
        )
        if self.verbose:
            print(
                "[policy] inference done | "
                f"arms={list(action_by_arm.keys())} "
                f"inference_time={end_time - start_time:.6f}s "
                f"raw_action_shape={list(raw_action.shape)} "
                f"env_action_shape={list(env_action.shape)}"
            )
        return log_response, send_response

    def _load_checkpoint(self, ckpt_path: Path) -> Tuple[dict, object]:
        resolved_ckpt_path = self._resolve_checkpoint_path(ckpt_path)
        payload = torch.load(
            open(resolved_ckpt_path, "rb"),
            map_location="cpu",
            pickle_module=dill,
        )
        return payload, payload["cfg"]

    def _resolve_checkpoint_path(self, ckpt_path: Path) -> Path:
        if not ckpt_path.is_dir():
            return ckpt_path

        epoch = str(self.checkpoint_epoch).strip()
        if epoch.lower() == "latest":
            resolved_ckpt_path = ckpt_path / "latest.ckpt"
            if not resolved_ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt_path}")
            return resolved_ckpt_path

        if epoch.isdigit():
            matches = sorted(ckpt_path.glob(f"epoch={int(epoch):04d}-*.ckpt"))
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint found for epoch={int(epoch):04d} under {ckpt_path}"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple checkpoints found for epoch={int(epoch):04d}: {matches}"
                )
            return matches[0]

        raise ValueError(
            f"Unsupported checkpoint epoch: {self.checkpoint_epoch}. "
            "Use an integer epoch like 10/20/40 or 'latest'."
        )

    def _create_policy(self, cfg, payload: dict, device: torch.device):
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.num_inference_steps = 16
        policy.eval().to(device)
        policy.reset()
        return policy

    def _validate_arm_mode_against_checkpoint(self) -> None:
        expected_robot_count = 1 if self.arm_mode == "single" else 2
        if self.model_robot_count != expected_robot_count:
            raise ValueError(
                "Checkpoint robot count does not match deployment mode. "
                f"Loaded checkpoint expects {self.model_robot_count} robot(s), "
                f"but arm_mode={self.arm_mode!r} expects {expected_robot_count}."
            )

    def _resolve_arm_payloads(
        self,
        payload: Dict,
        requested_arm: Optional[str] = None,
    ) -> List[Tuple[str, Dict]]:
        if self.arm_mode == "single":
            selected_arm, arm_payload = self._select_single_arm_payload(
                payload=payload,
                arm=requested_arm or self.preferred_arm,
            )
            return [(selected_arm, arm_payload)]
        return self._select_bimanual_payloads(payload)

    def _select_single_arm_payload(self, payload: Dict, arm: str) -> Tuple[str, Dict]:
        if self._is_flat_arm_payload(payload):
            return arm, payload

        available_arms = self._get_available_payload_arms(payload)
        if not available_arms:
            raise ValueError(
                "Payload format is not recognized. Expected flat keys "
                "(`images`, `poses`, `grippers`) or nested `arm_l` / `arm_r`."
            )
        if len(available_arms) == 1:
            selected_arm = available_arms[0]
            if self.verbose:
                print(f"[policy] only one arm found, automatically using `{selected_arm}`")
            return selected_arm, payload[selected_arm]
        if arm not in payload:
            raise ValueError(
                f"Requested arm `{arm}` not found in payload. Available arms: {available_arms}"
            )
        return arm, payload[arm]

    def _select_bimanual_payloads(self, payload: Dict) -> List[Tuple[str, Dict]]:
        if self._is_flat_arm_payload(payload):
            raise ValueError(
                "Bimanual deployment requires nested `arm_l` and `arm_r` payloads."
            )
        missing_arms = [
            arm_name
            for arm_name in ("arm_l", "arm_r")
            if not isinstance(payload.get(arm_name), dict)
        ]
        if missing_arms:
            raise ValueError(
                "Bimanual deployment requires both arms in payload. "
                f"Missing: {missing_arms}"
            )
        return [("arm_l", payload["arm_l"]), ("arm_r", payload["arm_r"])]

    def _build_env_obs(
        self,
        arm_payload_items: List[Tuple[str, Dict]],
        shape_meta: dict,
        img_save: bool = False,
    ) -> Tuple[Dict, List[np.ndarray]]:
        obs_shape_meta = shape_meta["obs"]
        env_obs: Dict[str, np.ndarray] = {}
        episode_start_pose: List[np.ndarray] = []

        if self.arm_mode == "single":
            arm_name, arm_payload = arm_payload_items[0]
            start_pose = self._populate_robot_obs(
                env_obs=env_obs,
                obs_shape_meta=obs_shape_meta,
                robot_idx=0,
                arm_name=arm_name,
                arm_payload=arm_payload,
                image_keys=self.rgb_keys,
                img_save=img_save,
            )
            episode_start_pose.append(start_pose)
            return env_obs, episode_start_pose

        if len(self.rgb_keys) != len(arm_payload_items):
            raise ValueError(
                "Bimanual deployment expects one RGB key per arm. "
                f"shape_meta rgb keys={self.rgb_keys}, payload arms={len(arm_payload_items)}"
            )

        for robot_idx, (arm_name, arm_payload) in enumerate(arm_payload_items):
            start_pose = self._populate_robot_obs(
                env_obs=env_obs,
                obs_shape_meta=obs_shape_meta,
                robot_idx=robot_idx,
                arm_name=arm_name,
                arm_payload=arm_payload,
                image_keys=[self.rgb_keys[robot_idx]],
                img_save=img_save,
            )
            episode_start_pose.append(start_pose)
        return env_obs, episode_start_pose

    def _populate_robot_obs(
        self,
        env_obs: Dict[str, np.ndarray],
        obs_shape_meta: Dict,
        robot_idx: int,
        arm_name: str,
        arm_payload: Dict,
        image_keys: List[str],
        img_save: bool,
    ) -> np.ndarray:
        decoded_images = [
            self._decode_rgb_image(image_b64) for image_b64 in arm_payload.get("images", [])
        ]
        if img_save:
            self._save_decoded_images(decoded_images, arm_name)

        poses = [
            self._pose7_to_pos_axis_angle(np.asarray(pose))
            for pose in arm_payload.get("poses", [])
        ]
        grippers = [float(x) for x in arm_payload.get("grippers", [])]
        if not poses:
            raise ValueError(f"Payload for {arm_name} does not contain any pose history.")
        if not grippers:
            raise ValueError(f"Payload for {arm_name} does not contain any gripper history.")

        pose_hist = np.stack(poses, axis=0).astype(np.float32)
        for image_key in image_keys:
            if image_key not in obs_shape_meta:
                raise ValueError(f"RGB key `{image_key}` is missing from shape_meta.")
            env_obs[image_key] = self._prepare_image_history(
                decoded_images=decoded_images,
                image_key=image_key,
                obs_shape_meta=obs_shape_meta,
                arm_name=arm_name,
                img_save=img_save,
            )

        init_pose_raw = arm_payload.get("init_pose")
        if init_pose_raw is None:
            init_pose = poses[0].copy()
        else:
            init_pose = self._pose7_to_pos_axis_angle(np.asarray(init_pose_raw))

        pos_key = f"robot{robot_idx}_eef_pos"
        rot_key = f"robot{robot_idx}_eef_rot_axis_angle"
        grip_key = f"robot{robot_idx}_gripper_width"

        if pos_key in obs_shape_meta:
            horizon = int(obs_shape_meta[pos_key]["horizon"])
            if len(pose_hist) < horizon:
                raise ValueError(
                    f"Payload for {arm_name} only has {len(pose_hist)} pose frames, "
                    f"but {pos_key} requires horizon={horizon}."
                )
            env_obs[pos_key] = pose_hist[-horizon:, :3]
        if rot_key in obs_shape_meta:
            horizon = int(obs_shape_meta[rot_key]["horizon"])
            if len(pose_hist) < horizon:
                raise ValueError(
                    f"Payload for {arm_name} only has {len(pose_hist)} pose frames, "
                    f"but {rot_key} requires horizon={horizon}."
                )
            env_obs[rot_key] = pose_hist[-horizon:, 3:]
        if grip_key in obs_shape_meta:
            horizon = int(obs_shape_meta[grip_key]["horizon"])
            if len(grippers) < horizon:
                raise ValueError(
                    f"Payload for {arm_name} only has {len(grippers)} gripper frames, "
                    f"but {grip_key} requires horizon={horizon}."
                )
            env_obs[grip_key] = np.asarray(grippers[-horizon:], dtype=np.float32)[:, None]

        return init_pose.astype(np.float32)

    def _prepare_image_history(
        self,
        decoded_images: List[np.ndarray],
        image_key: str,
        obs_shape_meta: Dict,
        arm_name: str,
        img_save: bool,
    ) -> np.ndarray:
        if not decoded_images:
            raise ValueError(f"Payload for {arm_name} does not contain any images.")
        horizon = int(obs_shape_meta[image_key]["horizon"])
        if len(decoded_images) < horizon:
            raise ValueError(
                f"Payload for {arm_name} only has {len(decoded_images)} image frames, "
                f"but {image_key} requires horizon={horizon}."
            )
        channels, height, width = obs_shape_meta[image_key]["shape"]
        if channels != 3:
            raise ValueError(
                f"Expected 3-channel RGB image for {image_key}, "
                f"got shape_meta={obs_shape_meta[image_key]['shape']}"
            )
        image_hist = decoded_images[-horizon:]
        image_shapes = [img.shape for img in image_hist]
        if len(set(image_shapes)) != 1:
            raise ValueError(f"Inconsistent image shapes in history for {image_key}: {image_shapes}")
        processed_images = [
            self._crop_resize_rgb_image(
                image_rgb=img,
                output_size=(width, height),
                crop=self.crop,
            )
            for img in image_hist
        ]
        if img_save:
            self._save_processed_images(
                processed_images,
                arm_name=arm_name,
                image_key=image_key,
            )
        return np.stack(processed_images, axis=0)

    def _split_env_action_by_arm(
        self,
        env_action: np.ndarray,
        arm_names: List[str],
    ) -> Dict[str, List[List[float]]]:
        action_by_arm: Dict[str, List[List[float]]] = {}
        for robot_idx, arm_name in enumerate(arm_names):
            start = robot_idx * 7
            end = start + 7
            if env_action.shape[-1] < end:
                raise ValueError(
                    f"env_action shape {env_action.shape} does not contain robot{robot_idx} output."
                )
            action_by_arm[arm_name] = self._env_action_to_xyz_quat_gripper(
                env_action[:, start:end]
            )
        return action_by_arm

    def _build_action_response(
        self,
        action_by_arm: Dict[str, List[List[float]]],
        timestamp: Optional[float] = None,
    ) -> Dict:
        arm_names = [arm_name for arm_name in ("arm_l", "arm_r") if arm_name in action_by_arm]
        return {
            "type": "action",
            "arm_mode": self.arm_mode,
            "arms": arm_names,
            "action_l": action_by_arm.get("arm_l", []),
            "action_r": action_by_arm.get("arm_r", []),
            "timestamp": time.time() if timestamp is None else timestamp,
        }

    def _save_decoded_images(self, decoded_images: List[np.ndarray], arm_name: str) -> None:
        if not decoded_images:
            return
        save_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_input_images"
            / time.strftime("%Y%m%d")
            / f"{arm_name}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, image_rgb in enumerate(decoded_images):
            image_path = save_dir / f"image_{frame_idx:03d}.png"
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(image_path), image_bgr)
            if not success:
                raise ValueError(f"Failed to save decoded image to {image_path}")

    def _save_processed_images(
        self,
        processed_images: List[np.ndarray],
        arm_name: str,
        image_key: str,
    ) -> None:
        if not processed_images:
            return
        save_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_processed_images"
            / time.strftime("%Y%m%d")
            / f"{arm_name}_{image_key}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, image_rgb in enumerate(processed_images):
            image_path = save_dir / f"image_{frame_idx:03d}.png"
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(image_path), image_bgr)
            if not success:
                raise ValueError(f"Failed to save processed image to {image_path}")

    def _save_policy_output(
        self,
        raw_action: np.ndarray,
        env_action: np.ndarray,
        action_by_arm: Dict[str, List[List[float]]],
    ) -> None:
        date_dir = (
            Path(__file__).resolve().parent
            / "output"
            / "policy_output"
            / time.strftime("%Y%m%d")
        )
        date_dir.mkdir(parents=True, exist_ok=True)
        file_path = date_dir / f"{self.arm_mode}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.json"
        file_path.write_text(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "arm_mode": self.arm_mode,
                    "raw_action": raw_action.tolist(),
                    "env_action": env_action.tolist(),
                    "action_by_arm": action_by_arm,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _env_action_to_xyz_quat_gripper(self, env_action: np.ndarray) -> List[List[float]]:
        actions: List[List[float]] = []
        for step in env_action:
            pos = step[:3]
            rotvec = step[3:6]
            grip = step[6]
            quat_xyzw = st.Rotation.from_rotvec(rotvec).as_quat()
            actions.append(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                    float(quat_xyzw[3]),
                    float(grip),
                ]
            )
        return actions

    def _apply_height_offset(self, actions: List[List[float]]) -> List[List[float]]:
        if not self.add_height:
            return actions

        offset_actions: List[List[float]] = []
        for step in actions:
            offset_step = list(step)
            offset_step[2] = float(offset_step[2]) + self.height
            offset_actions.append(offset_step)
        return offset_actions

    @staticmethod
    def _is_flat_arm_payload(payload: Dict) -> bool:
        return "images" in payload and "poses" in payload and "grippers" in payload

    @staticmethod
    def _get_available_payload_arms(payload: Dict) -> List[str]:
        return [
            arm_name
            for arm_name in ("arm_l", "arm_r")
            if isinstance(payload.get(arm_name), dict)
        ]

    @staticmethod
    def _normalize_arm_mode(arm_mode: str) -> str:
        normalized = str(arm_mode).strip().lower()
        if normalized not in {"single", "bimanual"}:
            raise ValueError(
                f"Unsupported arm_mode={arm_mode!r}. Use 'single' or 'bimanual'."
            )
        return normalized

    @staticmethod
    def _normalize_robot_transform(
        robot1_to_robot0_tx: Optional[Sequence[Sequence[float]]],
    ) -> np.ndarray:
        if robot1_to_robot0_tx is None:
            return np.eye(4, dtype=np.float32)
        tx = np.asarray(robot1_to_robot0_tx, dtype=np.float32)
        if tx.shape != (4, 4):
            raise ValueError(
                f"Expected robot1_to_robot0_tx shape (4, 4), got {tx.shape}"
            )
        return tx

    @staticmethod
    def _infer_robot_count(shape_meta: Dict) -> int:
        obs_shape_meta = shape_meta.get("obs", {})
        robot_indices = set()
        for key in obs_shape_meta.keys():
            prefix = key.split("_", 1)[0]
            if prefix.startswith("robot") and prefix[5:].isdigit():
                robot_indices.add(int(prefix[5:]))

        robot_count_from_obs = max(robot_indices) + 1 if robot_indices else 0
        action_shape = shape_meta.get("action", {}).get("shape", [])
        robot_count_from_action = 0
        if action_shape:
            action_dim = int(action_shape[0])
            if action_dim % 10 != 0:
                raise ValueError(
                    f"Expected action dim to be divisible by 10, got {action_dim}"
                )
            robot_count_from_action = action_dim // 10

        if robot_count_from_obs and robot_count_from_action:
            if robot_count_from_obs != robot_count_from_action:
                raise ValueError(
                    "shape_meta robot count is inconsistent between obs and action. "
                    f"obs={robot_count_from_obs}, action={robot_count_from_action}"
                )
        return robot_count_from_action or robot_count_from_obs or 1

    @staticmethod
    def _uses_episode_start_pose(shape_meta: Dict) -> bool:
        obs_shape_meta = shape_meta.get("obs", {})
        return any(key.endswith("_wrt_start") for key in obs_shape_meta.keys())

    @staticmethod
    def _sorted_obs_keys(obs_shape_meta: Dict, obs_type: str) -> List[str]:
        matching_keys = [
            key
            for key, attr in obs_shape_meta.items()
            if attr.get("type", "low_dim") == obs_type
        ]
        return sorted(matching_keys, key=PolicyInference._obs_key_sort_key)

    @staticmethod
    def _obs_key_sort_key(key: str) -> Tuple[int, str]:
        prefix = key.split("_", 1)[0]
        digits = "".join(ch for ch in prefix if ch.isdigit())
        index = int(digits) if digits else 0
        return index, key

    @staticmethod
    def _decode_rgb_image(image_b64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_b64, validate=True)
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("Failed to decode base64 JPEG image.")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    @staticmethod
    def _crop_resize_rgb_image(
        image_rgb: np.ndarray,
        output_size: Tuple[int, int],
        crop: bool,
    ) -> np.ndarray:
        if not crop:
            resized = cv2.resize(
                image_rgb,
                output_size,
                interpolation=cv2.INTER_AREA,
            )
            return resized.astype(np.uint8)

        height, width = image_rgb.shape[:2]
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        cropped = image_rgb[top : top + crop_size, left : left + crop_size]
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    @staticmethod
    def _pose7_to_pos_axis_angle(pose7: np.ndarray) -> np.ndarray:
        pose7 = np.asarray(pose7, dtype=np.float32)
        if pose7.shape != (7,):
            raise ValueError(f"Expected pose shape (7,), got {pose7.shape}")
        pos = pose7[:3]
        quat_xyzw = pose7[3:]
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
        return np.concatenate([pos, rotvec], axis=0)
