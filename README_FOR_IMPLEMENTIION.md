

## 移植流程

#### 1. 数据格式转换
把 `lerobot` 数据格式转换为 `diffusion_policy` 的 `zarr` 格式
#### 2. 修改训练配置
#####  2.1 修改task配置文件
在目录`universal_manipulation_interface/diffusion_policy/config/task`下 参考`cloth.yaml` 增加一个新的任务配置文件
#####  2.2 修改training配置文件
在目录`universal_manipulation_interface/diffusion_policy/config`下参考`train_diffusion_transformer_cloth_workspace.yaml` 增加一个训练配置文件
#### 3. 训练命令
##### 3.1 单卡训练
```bash
python3 train.py --config-name=train_diffusion_transformer_cloth_workspace task=cloth
```
##### 3.2 多卡训练
```bash
CUDA_VISIBLE_DEVICES=3,5,7 accelerate launch --num_processes 3 --mixed_precision bf16 train.py --config-name=train_diffusion_transformer_cloth_workspace task=cloth
```
#### 4. 远程推理
##### 4.1 推理部署
参考 `universal_manipulation_interface/realworld_deploy` 目录下的兼容双臂和单臂的推理代码`policy_inference.py`文件
##### 4.2 远程通信
参考 `universal_manipulation_interface/realworld_deploy` 目录下的`server.py`文件
