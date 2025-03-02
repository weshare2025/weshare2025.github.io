# 🚧 基于YOLOV8的道路病害检测

## 🔧 前期需求

- **开发环境**: PyCharm + Anaconda
- **YOLOV8项目地址**: [GitHub - ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
  - 使用命令克隆项目: `git clone https://github.com/ultralytics/ultralytics`

## 📦 环境需求

进入`ultralytics`目录并安装:

```bash
cd ultralytics
pip install -e .
```

## 🏗️ 架构变化

1. **Backbone**:
   - 第一层卷积由6x6卷积改为3x3卷积。
   - 将c3模块换成c2f模块，并调整模块深度。
2. **Neck**:
   - 移除1x1卷积的降通道层。
   - 将原本的c3模块换成c2f模块。
3. **Head**:
   - 换成解耦头结构，将分类任务和回归任务解耦。
   - 将Anchor-Based换成Anchor-Free。
4. **Loss**:
   - 使用BCE LOSS作为分类损失。
   - 使用DFL Loss + cIoU Loss作为回归损失。
5. **样本匹配策略**:
   - 采用Task-Aligned Assigner样本分配策略。
6. **训练策略**:
   - 新增最后10轮关闭Mosaic数据增强操作，提升精度。

| ![yolov8](https://weshare.xin/全球人工智能算法精英大赛/算法赛道[道路病害检测]/国家二等奖/yolov8.jpg) |
|:--:|
| **参考文档：MMYOLO > YOLOv8原理和实现全解析** |

## 📋 操作流程

### 📊 数据集

![数据分布](https://weshare.xin/全球人工智能算法精英大赛/算法赛道[道路病害检测]/国家三等奖/shujuji.jpg)

新建数据集yaml文件，示例如下：

```yaml
# yolov8_dataset
train: D:\\Pycharm_Projects\\ultralytics\\ultralytics\\datasets\\yolov8_dataset\\train # train images (relative to 'path') 128 images 
val: D:\\Pycharm_Projects\\ultralytics\\ultralytics\\datasets\\yolov8_dataset\\valid # val images (relative to 'path') 128 images 
test: D:\\Pycharm_Projects\\ultralytics\\ultralytics\\datasets\\yolov8_dataset\\test # test images (optional) 
# Classes 
names: 
 0: exam_1
 1: exam_2
 2: exam_3
 ...
```

### 🐍 Python指令训练

```python
from ultralytics import YOLO 

# Load a model # 三选一 
model = YOLO('yolov8n.yaml') # build a new model from YAML 
model = YOLO('yolov8n.pt') # load a pretrained model (recommended for training) 
model = YOLO('yolov8n.yaml').load('yolov8n.pt') # build from YAML and transfer weights 

# Train the model 
model.train(data='coco128.yaml', epochs=100, imgsz=640) 
```

### 🧪 验证模型

```python
from ultralytics import YOLO 

# Load a model
model = YOLO('yolov8n.pt') # load an official model 
model = YOLO('path/to/best.pt') # load a custom model 

# Validate the model 
metrics = model.val() # no arguments needed, dataset and settings remembered 
metrics.box.map # map50-95 
metrics.box.map50 # map50 
metrics.box.map75 # map75 
metrics.box.maps # a list contains map50-95 of each category
```

![数据分布](https://weshare.xin/全球人工智能算法精英大赛/算法赛道[道路病害检测]/国家三等奖/zhibiao.jpg)
![数据分布](https://weshare.xin/全球人工智能算法精英大赛/算法赛道[道路病害检测]/国家三等奖/jieguo.jpg)

### 🔍 预测模型

```python
from ultralytics import YOLO 

# Load a model 
model = YOLO('yolov8n.pt') # load an official model 
model = YOLO('path/to/best.pt') # load a custom model 

# Predict with the model 
results = model('https://ultralytics.com/images/bus.jpg') # predict on an image
```

![数据分布](https://weshare.xin/全球人工智能算法精英大赛/算法赛道[道路病害检测]/国家三等奖/shili.jpg)

### 📤 导出模型

```python
from ultralytics import YOLO 

# Load a model 
model = YOLO('yolov8n.pt') # load an official model 
model = YOLO('path/to/best.pt') # load a custom trained 

# Export the model 
model.export(format='onnx')
```

## 📈 数据集优化

### 📚 利用开源数据集（自主标注）

1. **确定标注格式**: YOLOv8采用“<object-class-id> <x> <y> <width> <height>”格式标注数据。标注前需明确标注规则，如目标框完整框住物体、不重叠、坐标为正等。
2. **选择标注工具**: 常用标注工具包括LabelImg、LabelMe、VIA等，推荐在线工具Make Sense，无需安装，支持多种标签类型和输出格式。

### 🖼️ 数据增强生成

1. **常用几何变换方法**: 翻转，旋转，裁剪，缩放，平移，抖动。使用时需注意标签数据的变化，如目标检测中翻转需调整gt框。
2. **常用像素变换方法**: 加椒盐噪声，高斯噪声，进行高斯模糊，调整HSV对比度，调节亮度、饱和度，直方图均衡化，调整白平衡等。
3. **其他数据增强方式**:
   - **Mixup**: 将随机的两张样本按比例混合，分类结果按比例分配，只适合分类任务。
   - **Cutout**: 随机将样本中的部分区域cut掉，填充0像素值，分类结果不变。
   - **Cutmix**: 将一部分区域cut掉但不填充0像素，而是随机填充训练集中其他数据的区域像素值，分类结果按一定比例分配。
   - **Mosaic**: 将4张图片按一定比例组合成一张图片。

## 🛠️ 模型优化

### 🔄 主干网络替换

使用Timm库融合1000+主干网络，如更换为FasterNet、VanillaNet、HGNetV2等，包括轻量化网络，以及双主干特征融合方式。以使用FasterNet替换主干网络为例：

1. 在`ultralytics/models/v8`文件夹下新建`yolov8-FasterNet.yaml`。
2. 将FasterNet核心代码添加到`ultralytics/nn/modules/block.py`文件末尾并修改。
3. 将相关类名加入到`ultralytics/nn/tasks.py`中。
4. 修改`yolov8-FasterNet.yaml`使用相关类构建FasterNet主干网络。
5. 开始训练。

### 🔍 添加注意力机制

在C2F模块等位置添加SE、CBAM、ECA等注意力机制。以在`yolov8.yaml`中添加SE注意力机制为例：

1. 在`ultralytics/models/v8`文件夹下新建`yolov8-sE.yaml`，拷贝`yolov8.yaml`内容。
2. 将SE注意力代码添加到`ultralytics/nn/modules/block.py`文件末尾，并在相关文件中添加SE。
3. 将SE类名加入到`ultralytics/nn/tasks.py`中。
4. 修改`yolov8-sE.yaml`，将SE注意力加到指定位置，修改相关系数。
5. 修改`ultralytics/yolo/cfg/default.yaml`文件的`-model`默认参数，添加`yolov8-sE.yaml`路径，开始训练。

### 🔗 特征融合改进

应用CARAFE、全维动态卷积、BiFPN结构等，引入EVC模块、AFPN结构等。以添加CARAFE为例：

1. 在`ultralytics/models/v8`文件夹下新建`yolov8-CARAFE.yaml`。
2. 将CARAFE代码添加到`ultralytics/nn/modules.py`文件末尾。
3. 将CARAFE类名加入到`ultralytics/nn/tasks.py`中。
4. 修改`yolov8-CARAFE.yaml`，使用CARAFE构建上采样模块。
5. 开始训练。

### ⚖️ 损失函数更换

将损失函数更换为CIoU、DIoU、EIoU等，引入MPDIoU、Shape-IoU等新损失函数。

### 🎯 改进检测头

更换挤压激励增强精准头、SEResNeXtBottleneck头、光晕自注意力Halo头，添加大目标和小目标检测头。

### 🆕 引入新模块和优化器

引入谷歌Lion优化器，更换20多种激活函数，如ReLU、LeakyReLU等，探索不同优化器和激活函数对模型的影响。

### 🔍 超参数调优与验证

利用Ray Tune进行超参数调优，实现K折交叉验证，解决数据集样本稀少和类别不平衡问题。
