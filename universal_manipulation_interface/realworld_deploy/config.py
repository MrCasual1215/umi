# Network
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007

SOCKET_TIMEOUT_SEC = 100.0
BUFFER_SIZE = 4096
ENCODING = "utf-8"
MAX_CLIENTS = 1


## ------------------------------------- Policy inference ------------------------------------
TASK = "green apple" # "green apple"
CAMERA = "rgb" # "rgb"
EPOCH = "latest"  # "latest"
CROP = True   
POLICY_MODEL = "unet_timm"  # "unet_timm" or "transformer"
ADD_HEIGHT = False
HEIGHT = 0.0 # meters



DEFAULT_POLICY_ARM = "arm_l"

# Logging / saving
BOOL = True
VERBOSE = BOOL     ## Whether to print verbose messages.
DATA_SAVE = BOOL     ## Whether to save payload records.
PICT_SAVE = BOOL ## Whether to save payload records.

# policy inference
ACTION_CHUNK_HORIZON = 16 ## Keep the first N actions from the predicted action chunk




POLICY_CONFIGS = [
    # {
    #     "task": "red block",
    #     "camera": "fisheye",
    #     "crop": False,
    #     "model": "unet_timm",
    #     "train_episode_count": 100,
    #     "epochs": {"latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.28/08.42.57_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },  # 100 episode data, fisheye + no crop, 60 epoch
    # {
    #     "task": "red block",
    #     "camera": "fisheye",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 100,
    #     "epochs": {"latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.28/14.36.42_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },  # 100 episode data, fisheye + crop, 40 epoch
    {
        "task": "red block",
        "camera": "fisheye",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 200,
        "epochs": {"10", "20", "30", "40", "50", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.28/20.24.37_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },  # 200 episode data, fisheye + crop, 59 epoch
    {
        "task": "red block",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 200,
        "epochs": {"10", "20", "30", "40", "50", "latest"},
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.29/07.27.29_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },  # 200 episode data, RGB + crop, 60 epoch
    # {
    #     "task": "green apple",
    #     "camera": "fisheye",
    #     "crop": True,
    #     "model": "unet_timm",
    #     "train_episode_count": 150,
    #     "epochs": {"10", "20", "30", "40", "50", "latest"},
    #     "checkpoint_path": (
    #         "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.29/18.59.34_train_diffusion_unet_timm_picknplace/checkpoints"
    #     ),
    # },  # 150 episode data, fisheye + crop, 60 epoch 待测试
        # {
        # "task": "green apple",
        # "camera": "rgb",
        # "crop": True,
        # "model": "unet_timm",
        # "train_episode_count": 150,
        # "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "latest"}, # 110 epoch
        # "checkpoint_path": (
        #     "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.30/09.03.52_train_diffusion_unet_timm_picknplace/checkpoints"
        # ),
        # },  # 150 episode data, rgb + crop, 110 epoch  待测试
# --------------------------------------------During Laboratory Day --------------------------------------------
    {
        "task": "green apple",
        "camera": "fisheye",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 300,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"}, # 120 epoch
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.01/16.44.26_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },  # 300 episode data, fisheye + crop, 120 epoch  待测试


    {
        "task": "green apple",
        "camera": "rgb",
        "crop": True,
        "model": "unet_timm",
        "train_episode_count": 300,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"}, # 120 epoch
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.03/10.52.21_train_diffusion_unet_timm_picknplace/checkpoints"
        ),
    },  # 300 episode data, rgb + crop, 120 epoch  待测试

        {
        "task": "green apple",
        "camera": "fisheye",
        "crop": True,
        "model": "transformer",
        "train_episode_count": 300,
        "epochs": {"10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "latest"}, # 200 epoch
        "checkpoint_path": (
            "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.05.04/21.34.50_train_diffusion_transformer_timm_picknplace/checkpoints"
        ),
    },  # 300 episode data, fisheye + crop, 200 epoch  待测试   
]


def _select_policy_checkpoint_path(task, camera, crop, epoch, model):
    task = task.strip().lower()
    camera = camera.strip().lower()
    epoch = str(epoch).strip().lower()
    model = model.strip().lower()

    matches = []
    for config in POLICY_CONFIGS:
        if (
            config["task"] == task
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
            f"task={task!r}, camera={camera!r}, crop={crop!r}, epoch={epoch!r}, model={model!r}. "
            f"Candidates: {match_desc}"
        )

    available_configs = ", ".join(
        f'{config["task"]} | {config["camera"]} | crop={config["crop"]} | '
        f'model={config["model"]} | epochs={sorted(config["epochs"])}'
        for config in POLICY_CONFIGS
    )
    raise ValueError(
        "No matching policy config for "
        f"TASK={TASK!r}, CAMERA={CAMERA!r}, CROP={CROP!r}, EPOCH={EPOCH!r}, POLICY_MODEL={POLICY_MODEL!r}. "
        f"Available configs: {available_configs}"
    )


POLICY_CHECKPOINT_PATH = _select_policy_checkpoint_path(TASK, CAMERA, CROP, EPOCH, POLICY_MODEL)
print(POLICY_CHECKPOINT_PATH)
