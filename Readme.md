# 明察芯毫——视觉智能体框架

<!-- <div align="center">
  <img src="pics/architecture.png" alt="AG-MCXH Architecture" width="800"/>
</div> -->

AG-MCXH（中文：明察芯毫）是一个基于大语言模型的视觉智能体框架，能够根据自然语言指令自动选择并调用多种视觉工具对图像进行分析和处理。该框架支持目标检测、图像分割、姿态估计、OCR等多种视觉任务。
> 一朵花开得最好的时候，就是屎吃的最多的时候。    ——《施肥》 ​​​

## 功能特点

- 🤖 **智能工具选择**: 基于自然语言指令自动选择最合适的视觉工具
- 🔧 **丰富的视觉工具**: 内置多种视觉处理工具，包括目标检测、图像分割等
- 🧠 **大模型集成**: 集成vLLM推理引擎，支持高性能推理
- 🌐 **Web界面**: 提供友好的Web用户界面
- 📦 **模块化设计**: 易于扩展的模型和工具注册机制

## 支持的视觉工具

### 目标检测
- YOLOv5/YOLOv8: 实时目标检测工具

### 图像分割
- SegmentAnything (SAM): 通用图像分割工具
- SegmentObject: 特定对象分割工具

### 图像处理
- OCR: 光学字符识别
- VQA: 视觉问答
- 人体姿态估计
- 人脸关键点检测
- Canny边缘检测
- 深度图生成
- 涂鸦草图生成

## 安装指南

### 环境要求
- Python 3.8+
- CUDA 11.8+ (用于GPU加速，可选)

### 安装步骤

```bash
git clone https://github.com/How-do-you-feel/Agent_MCXH.git
cd Agent_MCXH
pip install -r requirements.txt
```

#### 下载所需模型文件（根据需要选择）：
YOLO模型文件
SAM模型文件
大语言模型（如Qwen2.5系列

## 快速开始
### 直接调用
```python
from ag_mcxh.apis import load_tool

# 加载YOLO检测工具
yolo_tool = load_tool('YoloDetect',
                      model_path='/path/to/yolo11n.pt',
                      device='cpu',
                      conf_threshold=0.5)

# 处理图像
from ag_mcxh.types import ImageIO
image = ImageIO('/path/to/image.jpg')
detection_results = yolo_tool.apply(image)
print(detection_results)
```

## 扩展开发
### 注册新模型
在`ag_mcxh/models/`目录下创建模型实现，并通过装饰器或加载器函数注册到模型注册表。
### 添加新工具
在`ag_mcxh/tools/`目录下创建工具实现，继承BaseTool类并通过装饰器注册。

#### 详细开发指南请参考
- [模型注册指南](ag_mcxh/models/Readme.md)
- [工具开发指南](Agent_MCXH/ag_mcxh/tools/Readme.md)

### 使用实例
查看`ag_mcxh/examples/`目录中的示例代码：

- `example_yolo.py`: YOLO目标检测示例
- `vision_agent_example.py`: 视觉智能体使用示例
- `model_registration_example.py`: 模型注册示例

## 项目结构
```
Agent_MCXH/
├── ag_mcxh/              # 核心框架代码
│   ├── agent/            # 智能体实现
│   ├── apis/             # API接口
│   ├── models/           # 模型注册和管理
│   ├── tools/            # 视觉工具实现
│   ├── types/            # 数据类型定义
│   ├── utils/            # 工具函数
│   └── examples/         # 使用示例
├── webui/                # Web界面
├── scripts/              # 脚本工具
└── pics/                 # 图片资源
```
## 贡献
欢迎提交[Issue](https://github.com/How-do-you-feel/Agent_MCXH/issues)和[Pull Request](https://github.com/How-do-you-feel/Agent_MCXH/pulls)来改进本项目。
