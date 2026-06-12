# 部署流程

## 1. 数据集处理

### 1.1 移动数据集

将对齐之后的dataset移动到 umidata/single 目录下

```bash
cd umidata/single
scp yxgn@192.168.100.14:/home/yxgn/agilex/green_apple/data_150.zip ./
unzip data_150.zip
```

### 1.2 数据清洗

- 遥操
  - 检查每个episode中首帧位置是否异常 检查时间同步文件是否符合格式

```
python3 clean_and_analyze_umidata_single_arm_tele.py 
```

- umi 
  - 检查所有文件非空 并且有同步文件

```
python3 clean_and_analyze_umidata_single_arm_umi.py 
```

### 1.2 数据格式转换

修改 convert\_umidata\_single\_arm\_to\_umi\_zarr\_tele.py 参数

```YAML
CROP = True
FISHEYE = False
DATE = "20260428"
```

运行数据格式转换脚本

```bash
cd ../data_process
conda activate umi
python convert_umidata_single_arm_to_umi_zarr_tele.py
```

检查转换结果是否正确

```bash
python check_umi_zarr_dataset.py
```

<br />

## 2. 训练

#### 2.1 修改 picknplace.yaml 文件

```YAML
dataset_path: ../dataset/single/20260429_fisheye_croped_single.zarr.zip
```

#### 2.2 进行训练

```Shell
cd ~/sp/umi_project/universal_manipulation_interface/
conda activate umi
python3 train.py --config-name=train_diffusion_unet_timm_umi_workspace task=picknplace
```
```Shell
cd ~/sp/umi_project/universal_manipulation_interface/
conda activate umi
python3 train.py --config-name=train_diffusion_transformer_cloth_workspace task=cloth
```


#### 2.3 延时训练

```Shell
conda activate umi
bash train_script.sh 
```

## 3. 开环验证

#### 3.1 数据格式转换

将pika原始数据转换为实际推理用到的json格式

```bash
rm -r json_output/*
python episode2json.py \
  --episode-dir /home/sunpeng/sp/umi_project/umidata/single/20260601 \
  --start-index 90 \
  --end-index 100 \
  --camera-name pikaGripperDepthCamera \
  --pose-rel-path arm/endPose/sensorPose
```

#### 3.2 开环验证

对所有的json文件进行推理，进行开环验证

```bash
rm -r validation_output/episode*.json
conda activate umi
python validate_openloop_policy.py
```

<br />

## 4. 实物部署

#### 4.1 开启服务器

先在config中修改配置

```python
POLICY_CHECKPOINT_PATH = (
    "/home/sunpeng/sp/umi_project/universal_manipulation_interface/data/outputs/2026.04.29/18.59.34_train_diffusion_unet_timm_picknplace/checkpoints"
) 
CROP = True
EPOCH = "50"  ## 可选: 10 / 20 / 30 / 40 / 50 / "latest"

VERBOSE = True     ## Whether to print verbose messages.
PRINT = True     ## Whether to print payload records.
PICT_SAVE = True ## Whether to save payload records.
```

运行服务器

```bash
conda activate umi
python server.py
```

#### 4.2 可视化动作

导出整个目录的高分辨率 GIF：

```bash
python visual_action.py   --input  output/sent_actions/20260423   --output output/gif/fisheye_greenapple_crop/actions_with_obs.gif   --no-show   --fps 2   --width 16   --height 9   --dpi 220
```

删除output目录下的所有文件

```bash
bash clear.sh
```

