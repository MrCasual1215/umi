# UMI 项目使用说明

这个仓库是在原始 `universal_manipulation_interface` 基础上扩展的本地工作区，已经包含了以下完整链路：

1. 原始数据整理与清洗
2. 训练数据转换为 `zarr.zip`
3. 单臂 / 双臂策略训练
4. 开环离线验证
5. TCP 服务端远程推理部署
6. 动作可视化与结果落盘

如果你是第一次接手这个仓库，建议先读完本文件，再按对应阶段进入子目录操作。

## 1. 仓库结构

根目录主要分为以下几部分：

```text
umi_project/
├── calibration/                      # 标定相关脚本
├── umidata/                          # UMI 原始数据与转换脚本
│   └── data_process/
├── universal_manipulation_interface/ # 训练、验证、部署主代码
├── README.md
└── README_FOR_IMPLEMENTIION.md
```

其中最常用的目录如下：

- `umidata/data_process/`
  - UMI 数据清洗与 `zarr` 转换脚本
- `universal_manipulation_interface/diffusion_policy/config/`
  - 训练配置
- `universal_manipulation_interface/diffusion_policy/config/task/`
  - 任务配置，主要改 `dataset_path`
- `universal_manipulation_interface/openloop_validate/`
  - 开环验证脚本
- `universal_manipulation_interface/realworld_deploy/`
  - TCP 推理服务、部署配置、结果可视化

## 2. 环境准备

推荐使用 `umi` conda 环境。原始环境文件位于：

- `universal_manipulation_interface/conda_environment.yaml`

初始化方式：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface
mamba env create -f conda_environment.yaml
conda activate umi
```

如果环境已经存在，后续所有命令默认都在以下前提下执行：

```bash
conda activate umi
```

## 3. 常见工作流总览

### 3.1 UMI 单臂数据

```text
原始 episode -> 清洗检查 -> 转 zarr -> 修改 task 配置 -> train.py -> 开环验证/部署
```

## 4. 数据集处理

### 4.1 拷贝或放置原始数据

如果原始数据已经在其他机器上，可先同步到对应目录。例如单臂 UMI 数据常放在 `umidata/single/` 下：

```bash
cd /home/sunpeng/sp/umi_project/umidata/single
scp yxgn@192.168.100.14:/home/yxgn/agilex/green_apple/data_150.zip ./
unzip data_150.zip
```

建议整理后的目录风格保持一致，例如：

```text
umidata/
└── single/
    └── 20260428/
        ├── episode0/
        ├── episode1/
        └── ...
```

### 4.2 数据清洗

清洗脚本位于 `umidata/data_process/`。

#### 单臂遥操数据

用途：

- 检查每个 episode 首帧位置是否异常
- 检查同步文件格式是否正常

```bash
cd /home/sunpeng/sp/umi_project/umidata/data_process
conda activate umi
python3 clean_and_analyze_umidata_single_arm_tele.py
```

#### 单臂 UMI 数据

用途：

- 检查文件是否非空
- 检查是否存在同步文件

```bash
cd /home/sunpeng/sp/umi_project/umidata/data_process
conda activate umi
python3 clean_and_analyze_umidata_single_arm_umi.py
```

#### 双臂 UMI 数据

```bash
cd /home/sunpeng/sp/umi_project/umidata/data_process
conda activate umi
python3 clean_and_analyze_umidata_dual_arm_umi.py
```

### 4.3 UMI 数据转换为训练集

`umidata/data_process/` 下面目前有以下转换脚本：

- `convert_umidata_single_arm_to_umi_zarr_tele.py`
- `convert_umidata_single_arm_to_umi_zarr_umi.py`
- `convert_umidata_single_arm_to_umi_zarr_merged.py`
- `convert_umidata_dual_arm_to_umi_zarr_umi.py`

以单臂遥操数据为例，通常需要先修改脚本顶部参数，例如：

```python
CROP = True
FISHEYE = False
DATE = "20260428"
```

然后执行转换：

```bash
cd /home/sunpeng/sp/umi_project/umidata/data_process
conda activate umi
python3 convert_umidata_single_arm_to_umi_zarr_tele.py
```

如果你使用的是 UMI 原始数据或双臂数据，则切换到对应的 `convert_*` 脚本执行。

## 5. 训练

训练入口统一是：

- `universal_manipulation_interface/train.py`

训练前通常只需要做两件事：

1. 修改任务配置里的 `dataset_path`
2. 选择合适的训练配置 `--config-name`

### 5.1 常用任务配置

目录：

- `universal_manipulation_interface/diffusion_policy/config/task/`

当前常用配置包括：

- `picknplace.yaml`
  - 单臂任务
  - 当前默认数据集：`../lpc/dataset/single/multi_cam_record_umi_dp_cam2_right_single.zarr.zip`
- `cloth.yaml`
  - 双臂任务
  - 当前默认数据集：`../dataset/double/sweather_0630_plus_sweather_0701_RGB_croped_dual.zarr.zip`

修改示例：

```yaml
dataset_path: ../dataset/single/20260429_fisheye_croped_single.zarr.zip
```

### 5.2 常用训练配置

目录：

- `universal_manipulation_interface/diffusion_policy/config/`

当前最常用的两个配置：

- `train_diffusion_unet_timm_umi_workspace.yaml`
  - 常用于单臂任务
- `train_diffusion_transformer_cloth_workspace.yaml`
  - 常用于双臂 cloth / sweather 类任务

### 5.3 单臂训练

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface
conda activate umi
python3 train.py --config-name=train_diffusion_unet_timm_umi_workspace task=picknplace
```

### 5.4 双臂训练

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface
conda activate umi
python3 train.py --config-name=train_diffusion_transformer_cloth_workspace task=cloth
```

### 5.5 多卡训练

如果需要多卡，可使用 `accelerate`：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface
conda activate umi
CUDA_VISIBLE_DEVICES=3,5,7 accelerate launch --num_processes 3 --mixed_precision bf16 \
  train.py --config-name=train_diffusion_transformer_cloth_workspace task=cloth
```

### 5.6 延时训练

仓库内已有延时训练脚本：

- `universal_manipulation_interface/train_script.sh`

当前内容是等待 10 小时后启动一次单臂训练：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface
conda activate umi
bash train_script.sh
```

如果你要改成别的训练任务，直接编辑这个脚本即可。

### 5.7 训练输出位置

默认训练结果通常保存在：

```text
universal_manipulation_interface/data/outputs/<日期>/<时间>_<训练配置>_<任务名>/
```

重点关注：

- `checkpoints/`
- `logs.json.txt`
- 可视化与评估输出

## 6. 开环验证

开环验证位于：

- `universal_manipulation_interface/openloop_validate/`

包含两个主要脚本：

- `episode2json.py`
  - 把原始 episode 转成推理用 JSON
- `validate_openloop_policy.py`
  - 用部署侧 `PolicyInference` 做离线推理验证

### 6.1 原始数据转 JSON

将 Pika 原始数据转换为实际推理所需的 `json_output/episode*/sample_*` 结构。

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate
conda activate umi
python3 episode2json.py \
  --episode-dir /home/sunpeng/sp/umi_project/umidata/single/20260601 \
  --start-index 90 \
  --end-index 100 \
  --camera-name pikaGripperDepthCamera \
  --pose-rel-path arm/endPose/sensorPose
```

常用参数说明：

- `--episode-dir`
  - 原始 episode 根目录，或者单个 episode 目录
- `--start-index` / `--end-index`
  - 处理的 episode 编号范围
- `--camera-name`
  - 例如 `pikaGripperDepthCamera` 或 `pikaGripperFisheyeCamera`
- `--pose-rel-path`
  - 指定姿态路径，例如 `arm/endPose/sensorPose`
- `--output-dir`
  - 默认输出到 `openloop_validate/json_output`

如果要清空旧结果后重新生成：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate
rm -rf json_output/*
```

### 6.2 运行开环验证

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate
conda activate umi
python3 validate_openloop_policy.py
```

这个脚本会默认读取：

- 输入目录：`json_output/`
- 输出目录：`validation_output/`
- checkpoint 与 crop 等部署参数：自动从 `../realworld_deploy/config.py` 读取

因此在跑开环验证前，最好先确认部署配置是正确的。

如果只想验证部分 episode 或指定 checkpoint，可使用：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate
conda activate umi
python3 validate_openloop_policy.py \
  --episode episode90 \
  --episode episode91 \
  --checkpoint-path /home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/your_run/checkpoints \
  --checkpoint-epoch latest
```

如果要清理旧结果：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/openloop_validate
rm -rf validation_output/*
```

## 7. 实物部署

部署相关代码位于：

- `universal_manipulation_interface/realworld_deploy/`

核心文件：

- `config.py`
  - 部署配置中心
- `policy_inference.py`
  - 加载 checkpoint 并做策略推理
- `server.py`
  - TCP 服务端
- `CLIENT_API.md`
  - 客户端通信协议说明
- `visual_action.py`
  - 动作可视化与 GIF 导出

### 7.1 先修改部署配置

部署前请先编辑：

- `universal_manipulation_interface/realworld_deploy/config.py`

当前这个文件不是简单写死单个 `POLICY_CHECKPOINT_PATH`，而是通过以下字段自动匹配：

```python
TASK = "sweather"
MODE = "umi"              # "umi" or "tele"
CAMERA = "fisheye"        # "rgb" or "fisheye"
EPOCH = "80"              # "latest" or a specific epoch string
CROP = True
POLICY_MODEL = "transformer"   # "unet_timm" or "transformer"
POLICY_ARM_MODE = "bimanual"   # "single" or "bimanual"
ADD_HEIGHT = False
HEIGHT = 0.0
```

并在 `POLICY_CONFIGS` 中找到与你当前任务匹配的配置项。

### 7.2 启动服务器

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/realworld_deploy
conda activate umi
python3 server.py
```

当前默认网络配置在 `config.py` 中：

```python
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8007
```

### 7.3 客户端通信协议

客户端协议文档：

- `universal_manipulation_interface/realworld_deploy/CLIENT_API.md`

当前服务使用：

- TCP
- `utf-8`
- 按行分隔 JSON
- 每条消息必须以 `\n` 结尾

支持的请求类型：

- `observation`
- `reset`

### 7.4 动作可视化

导出整个目录的高分辨率 GIF：

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/realworld_deploy
conda activate umi
python3 visual_action.py \
  --input output/sent_actions/20260423 \
  --output output/gif/fisheye_greenapple_crop/actions_with_obs.gif \
  --no-show \
  --fps 2 \
  --width 16 \
  --height 9 \
  --dpi 220
```

### 7.5 清理部署输出

```bash
cd /home/sunpeng/sp/umi_project/universal_manipulation_interface/realworld_deploy
bash clear.sh
```

## 8. 手眼标定

#### 8.1 为什么需要标定

机器人的坐标系与umi数据采集时的坐标系没有对齐 

#### 8.2 标定流程

1. 打印标定图片 位于`calibration/tag.pdf`
2. 将标定图片放置在桌umi桌面的右上角 注意一定要与桌面的横竖对齐
3. 单臂采集一条大概30s左右的数据 要求采集去过程中有较大的旋转和平移的变化 并且图像中包含tag
4. 获取基站的baselink到tag系的转换以及sensor到相机系的转换
5. 将这个转换填入工控机的`/home/yxgn/pika_ros/src/sensor_tools/scripts/tag_pose_transform_dual.py 和 tag_pose_transform.py 中`

## 9. 原始 UMI 子仓库说明

如果你需要了解原始框架的安装、SLAM 流程、官方真实机器人评估方式，请继续参考：

- `universal_manipulation_interface/README.md`

这个文件更偏向原始 UMI 官方流程，包括：

- 环境安装
- SLAM pipeline
- 官方训练样例
- 官方实机评估样例

## 10. 二次开发说明

如果你的目标不是直接训练当前任务，而是把新的数据格式或新机器人流程接入进来，请继续看：

- `README_FOR_IMPLEMENTIION.md`

这个文件更偏向：

- 如何新增 task 配置
- 如何新增 training 配置
- 如何接入远程推理
- 如何改远程通信

## 11. 常见修改点速查

### 11.1 改训练数据路径

编辑：

- `universal_manipulation_interface/diffusion_policy/config/task/picknplace.yaml`
- `universal_manipulation_interface/diffusion_policy/config/task/cloth.yaml`

重点字段：

```yaml
dataset_path: ...
```

### 11.2 改部署 checkpoint

编辑：

- `universal_manipulation_interface/realworld_deploy/config.py`

重点字段：

- `TASK`
- `MODE`
- `CAMERA`
- `EPOCH`
- `CROP`
- `POLICY_MODEL`
- `POLICY_ARM_MODE`
- `POLICY_CONFIGS`

### 11.3 改开环验证输入输出

命令行参数：

- `episode2json.py --output-dir ...`
- `validate_openloop_policy.py --input-dir ... --output-dir ...`

## 12. 推荐使用顺序

### 单臂任务

1. 把原始数据整理到 `umidata/` 或 `lpc/umidata/`
2. 用对应脚本做清洗与转换
3. 修改 `picknplace.yaml` 的 `dataset_path`
4. 运行 `train_diffusion_unet_timm_umi_workspace`
5. 用 `openloop_validate/` 做开环验证
6. 修改 `realworld_deploy/config.py` 后启动 `server.py`

### 双臂任务

1. 把原始数据整理到 `umidata/` 或 `ldx/teledata/`
2. 用双臂转换脚本生成 `zarr.zip`
3. 修改 `cloth.yaml` 的 `dataset_path`
4. 运行 `train_diffusion_transformer_cloth_workspace`
5. 先做开环验证
6. 再切到 `realworld_deploy/` 做双臂部署

