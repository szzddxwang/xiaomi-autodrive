第二周的主要目标是学习 CARLA Python API 的基本使用方法，掌握如何通过 Python 脚本控制仿真车辆和虚拟传感器，并在 CARLA 城市道路场景中采集自动驾驶仿真数据。

本项目使用 CARLA 作为自动驾驶仿真平台，在 ego vehicle 上配置 RGB Camera 和 LiDAR 两类虚拟传感器，采集图像、点云、目标标注和相机标定文件，并将数据保存为 KITTI-like 数据集结构。最终生成 1000 帧仿真数据，为后续目标检测、车道线检测和自动驾驶感知算法开发提供数据基础。

## 1. 开发环境
操作系统：Windows + WSL2 Ubuntu 22.04

IDE：PyCharm

Python 版本：Python 3.10

python 虚拟环境：venv

CARLA 版本：CARLA 0.9.15

CARLA server 运行平台：Windows

Python client 运行平台：WSL2 Ubuntu 22.04

## 2. 项目文件夹内容

在上传到github 的文件里，各目录说明如下：


scripts：保存 Python 脚本

collect_kitti_dataset.py：CARLA 数据采集脚本

visualize_kitti_labels.py：KITTI 标注可视化脚本

dataset/image_2：保存 RGB 图像

dataset/label_2：保存 KITTI-like 标注文件

dataset/velodyne：保存 LiDAR 点云文件

dataset/calib：保存相机标定文件

docs：保存实验文档或截图说明


## 3. 第二周任务内容

第二周的任务如下：

1. 学习 CARLA Python API，编写脚本控制车辆和传感器。
2. 配置 RGB Camera 和 LiDAR 等虚拟传感器，采集仿真数据。
3. 学习 KITTI 数据集格式，并将 CARLA 仿真数据转换为 KITTI-like 格式。
4. 学习 Git 版本控制，初始化 GitHub 项目仓库并编写 README 文档。

## 4. 数据采集
在启动 CARLA 服务器以及激活 Python 环境后，可以使用数据采集程序完成任务

数据采集程序为：

```text
scripts/collect_kitti_dataset.py
```

该脚本主要完成以下功能：

1. 连接 Windows 端运行的 CARLA Server。
2. 创建主视角车辆（ego vehicle)，并开启自动驾驶模式。
3. 生成车辆，构建城市道路交通场景。
4. 在 ego vehicle 上安装 RGB Camera。
5. 在 ego vehicle 上安装 LiDAR。
6. 使用同步模式采集图像和点云数据。
7. 根据 CARLA actor 的 bounding box 自动生成 KITTI-like 标注文件。
8. 保存图像、标注、点云和标定文件。
9. 生成 1000 帧仿真数据。

## 5. 运行数据采集程序

进入脚本目录：

```bash
cd ~/真实路径/week2_data_collection/scripts
source ~/真实路径/venv/bin/activate
```

然后运行 1000 帧采集程序：

```bash
cd ~/真实路径/week2_data_collection/scripts
source ~/真实路径/venv/bin/activate
python collect_kitti_dataset.py --host 172.21.240.1 --port 2000 --frames 1000 --walkers 0 --vehicles 30
```
当采集完成后，可以关闭并重新启动 CARLA 以清理场景中的剩余 actor

## 6. 数据集格式说明

本项目生成的数据采用 KITTI-like 数据集结构，其中：

```text
image_2：保存 RGB Camera 图像
label_2：保存 KITTI-like 目标标注
velodyne：保存 LiDAR 点云数据
calib：保存相机内参和标定信息
```

在`label_2` 中每一行表示一个目标，格式如下：

```text
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y
```

即为：

```text
type：目标类别
truncated：目标截断程度
occluded：目标遮挡程度
alpha：目标观察角
bbox_left bbox_top bbox_right bbox_bottom：图像里的 2D 框坐标
height width length：目标 3D 尺寸
x y z：目标相对相机的位置
rotation_y：目标绕 y 轴旋转角
```

## 7. 类别标注说明

本项目的标注类别来自 CARLA actor 的 `type_id`，不是由深度学习模型预测得到

类别映射规则如下：

```text
普通四轮车辆: Car
行人: Pedestrian
两轮车: Cyclist
其他类型: other
```

## 8. 可视化标注检查

为了检查图像和 KITTI-like 标注是否匹配，可以运行可视化程序：

```text
scripts/visualize_kitti_labels.py
```

运行方式如下：

```bash
cd ~/真实路径/week2_data_collection/scripts
source ~/真实路径/venv/bin/activate
python visualize_kitti_labels.py
```

脚本会读取指定帧的图像和标注文件，并在图像上绘制 2D bounding box 和类别名称

然后生成三张可视化图片：

```text
dataset/vis_000000.png
dataset/vis_000100.png
dataset/vis_000500.png
```

这三张图用于展示不同时间点的标注效果，没有对全部 1000 帧进行可视化处理
最后将成果上传到 github 中

## 9. 第二周完成结果

本周完成了 CARLA Python API 的学习与基础使用，编写了 CARLA 数据采集脚本，并在 CARLA Town10HD_Opt 城市道路场景中采集了 1000 帧仿真数据。数据包括 RGB 图像、KITTI-like 标注文件、LiDAR 点云文件和相机标定文件

同时，本项目完成了 KITTI-like 数据格式转换，并生成了三张标注可视化结果图，用于检查 bounding box 和类别标注是否正确。最后，完成了 Git 仓库初始化，编写 README.md，并准备了 sample_data 用于 GitHub 展示

