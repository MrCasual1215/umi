# Network
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007

SOCKET_TIMEOUT_SEC = 100.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 1


## ------------------------------------- Policy inference ------------------------------------
TASK = "orange block" # "orange block"
MODE = "umi"  # "umi" or "tele"
CAMERA = "rgb" # "rgb"
EPOCH = "latest"  # "latest"
CROP = True   
POLICY_MODEL = "unet_timm"  # "unet_timm" or "transformer"
ADD_HEIGHT = False
HEIGHT = 0.0 # meters  



DEFAULT_POLICY_ARM = "arm_l"

# Logging / saving
BOOL = False
VERBOSE = BOOL     ## Whether to print verbose messages.
DATA_SAVE = BOOL     ## Whether to save payload records.
PICT_SAVE = BOOL ## Whether to save payload records.

# policy inference
ACTION_CHUNK_HORIZON = 16 ## Keep the first N actions from the predicted action chunk




POLICY_CONFIGS = [
# --------------------------------------------orange block -------------------------------------------- #

# --------------------------------------------umi -------------------------------------------- #
    # {
    #     "task": "orange block",
    #     "mode": "umi",
    #     "camera": "rgb",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 201,
    #     "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.13/20.45.39_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },

    # {
    #     "task": "orange block",
    #     "mode": "umi",
    #     "camera": "rgb",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 201,
    #     "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.28/11.05.29_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },

    {
        "task": "orange block",
        "mode": "umi",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 251,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.06.01/19.59.20_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },


    {
        "task": "orange block",
        "mode": "umi",
        "camera": "rgb",
        "crop": True,
        "model": "transformer",
        "train_episode_count": 603,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.29/14.57.20_train_diffusion_transformer_timm_picknplace/checkpoints"
        ),
    },
# --------------------------------------------tele -------------------------------------------- #

    {
        "task": "orange block",
        "mode": "tele",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 201,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.26/20.45.38_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },

]


def _select_policy_checkpoint_path(task, mode, camera, crop, epoch, model):
    task = task.strip().lower()
    mode = mode.strip().lower()
    camera = camera.strip().lower()
    epoch = str(epoch).strip().lower()
    model = model.strip().lower()

    matches = []
    for config in POLICY_CONFIGS:
        if (
            config["task"] == task
            and config["mode"] == mode
            and config["camera"] == camera
            and config["crop"] == crop
            and config["model"] == model
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
            f"task={task!r}, mode={mode!r}, camera={camera!r}, crop={crop!r}, epoch={epoch!r}, model={model!r}. "
            f"Candidates: {match_desc}"
        )

    available_configs = ", ".join(
        f'{config["task"]} | {config["mode"]} | {config["camera"]} | crop={config["crop"]} | '
        f'model={config["model"]} | epochs={sorted(config["epochs"])}'
        for config in POLICY_CONFIGS
    )
    raise ValueError(
        "No matching policy config for "
        f"TASK={TASK!r}, MODE={MODE!r}, CAMERA={CAMERA!r}, CROP={CROP!r}, EPOCH={EPOCH!r}, POLICY_MODEL={POLICY_MODEL!r}. "
        f"Available configs: {available_configs}"
    )


POLICY_CHECKPOINT_PATH = _select_policy_checkpoint_path(
    TASK, MODE, CAMERA, CROP, EPOCH, POLICY_MODEL
)
print(POLICY_CHECKPOINT_PATH)
