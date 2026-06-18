# Network
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007

SOCKET_TIMEOUT_SEC = 100.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 1


## ------------------------------------- Policy inference ------------------------------------
TASK = "orange block" # "orange block" or "cloth"
MODE = "tele"  # "umi" or "tele"
CAMERA = "rgb" # "rgb"
EPOCH = "latest"  # "latest"
CROP = True   
POLICY_MODEL = "unet_timm"  # "unet_timm" or "transformer"
POLICY_ARM_MODE = "single"  # "single" or "bimanual"
ADD_HEIGHT = False
HEIGHT = 0.0 # meters  



DEFAULT_POLICY_ARM = "arm_l"
# Relative transform from robot1 base frame to robot0 base frame.
# Keep identity when both arm poses are already expressed in the same world frame.
ROBOT1_TO_ROBOT0_TX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# Logging / saving
BOOL = True
VERBOSE = BOOL     ## Whether to print verbose messages.
DATA_SAVE = BOOL     ## Whether to save payload records.
PICT_SAVE = BOOL ## Whether to save payload records.

# policy inference
ACTION_CHUNK_HORIZON = 16 ## Keep the first N actions from the predicted action chunk




POLICY_CONFIGS = [
# --------------------------------------------orange block -------------------------------------------- #

# --------------------------------------------umi -------------------------------------------- #
  

    {
        "task": "orange block",
        "mode": "umi",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "arm_mode": "single",
        "train_episode_count": 251,
        "epochs": {"latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/a100_250_mixed/checkpoints"
        ),
    },  ## include left and right 

    {
        "task": "cloth",
        "mode": "umi",
        "camera": "rgb",
        "crop": True,
        "model": "transformer",
        "arm_mode": "bimanual",
        "train_episode_count": 251,
        "epochs": {"60", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/a100_dual_shorts/checkpoints"
        ),
    },

    {
        "task": "cloth",
        "mode": "umi",
        "camera": "fisheye",
        "crop": True,
        "model": "transformer",
        "arm_mode": "bimanual",
        "train_episode_count": 251,
        "epochs": {"latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/a100_dual_shorts_fisheye/checkpoints"
        ),
    },




    # {
    #     "task": "orange block",
    #     "mode": "umi",
    #     "camera": "rgb",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 648,
    #     "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.06.03/17.30.02_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },   ## include left and right 


    # {
    #     "task": "orange block",
    #     "mode": "umi",
    #     "camera": "rgb",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 196,
    #     "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.06.05/19.34.08_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },   ## left only





# --------------------------------------------tele -------------------------------------------- #

    {
        "task": "orange block",
        "mode": "tele",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "arm_mode": "single",
        "train_episode_count": 201,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.26/20.45.38_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },

]


def _normalize_policy_arm_mode(arm_mode):
    normalized = str(arm_mode).strip().lower()
    if normalized not in {"single", "bimanual"}:
        raise ValueError(
            f"Unsupported POLICY_ARM_MODE={arm_mode!r}. Use 'single' or 'bimanual'."
        )
    return normalized


def _select_policy_checkpoint_path(task, mode, camera, crop, epoch, model, arm_mode):
    task = task.strip().lower()
    mode = mode.strip().lower()
    camera = camera.strip().lower()
    epoch = str(epoch).strip().lower()
    model = model.strip().lower()
    arm_mode = _normalize_policy_arm_mode(arm_mode)

    matches = []
    for config in POLICY_CONFIGS:
        config_arm_mode = _normalize_policy_arm_mode(config.get("arm_mode", "single"))
        if (
            config["task"] == task
            and config["mode"] == mode
            and config["camera"] == camera
            and config["crop"] == crop
            and config["model"] == model
            and config_arm_mode == arm_mode
            and epoch in config["epochs"]
        ):
            matches.append(config)

    if len(matches) == 1:
        return matches[0]["checkpoint_path"]
    if len(matches) > 1:
        match_desc = ", ".join(
            f'{config["checkpoint_path"]} (episodes={config["train_episode_count"]})'
            for config in matches
        )
        raise ValueError(
            "Multiple matching policy configs found for "
            f"task={task!r}, mode={mode!r}, camera={camera!r}, crop={crop!r}, epoch={epoch!r}, model={model!r}, arm_mode={arm_mode!r}. "
            f"Candidates: {match_desc}"
        )

    available_configs = ", ".join(
        f'{config["task"]} | {config["mode"]} | {config["camera"]} | crop={config["crop"]} | '
        f'model={config["model"]} | arm_mode={config.get("arm_mode", "single")} | epochs={sorted(config["epochs"])}'
        for config in POLICY_CONFIGS
    )
    raise ValueError(
        "No matching policy config for "
        f"TASK={TASK!r}, MODE={MODE!r}, CAMERA={CAMERA!r}, CROP={CROP!r}, EPOCH={EPOCH!r}, POLICY_MODEL={POLICY_MODEL!r}, POLICY_ARM_MODE={POLICY_ARM_MODE!r}. "
        f"Available configs: {available_configs}"
    )


POLICY_CHECKPOINT_PATH = _select_policy_checkpoint_path(
    TASK, MODE, CAMERA, CROP, EPOCH, POLICY_MODEL, POLICY_ARM_MODE
)
print(POLICY_CHECKPOINT_PATH)
