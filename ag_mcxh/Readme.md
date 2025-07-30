# AG-MCXH 视觉智能体框架

AG-MCXH 是一个视觉工具框架，允许通过大语言模型调用多种视觉检测工具对上传的图片进行检测和分析。

## 安装

```bash
cd ag_mcxh
pip install -r requirements.txt
```

## 支持的工具

**图像处理相关**
- SegmentAnything - 图像分割工具
- SegmentObject - 特定对象分割工具
- YoloDetect - YOLO目标检测工具
- ImageDescription: 描述输入图像。
- OCR：从照片中识别文本。
- VQA：根据图片回答问题。
- OBB: 估计图像中人体的姿态或关键点，并绘制人体姿态图像
- HumanFaceLandmark: 识别图像中人脸的关键点，并绘制带有关键点的图像。
- ImageToCanny: 从图像中提取边缘图像。
- ImageToDepth: 生成图像的深度图像。
- ImageToScribble: 生成一张图像的涂鸦草图。
- ObjectDetection: 检测图像中的所有物体。
- TextToBbox: 检测图像中的给定对象。
- SegmentAnything: 分割图像中的所有物体。
- SegmentObject: 根据给定的物体名称，在图像中分割出特定的物体。


## API说明
### 1. 工具加载 (load_tool)
加载指定的视觉工具

```python
from ag_mcxh.apis import load_tool
tool = load_tool('工具名称', 参数...)
```

#### 参数：
- `tool_type (str)`: 工具名称
- `**kwargs`: 工具特定参数
#### 示例
```python
# 加载YOLO检测工具
yolo_tool = load_tool('YoloDetect',
                      model_path='/path/to/yolo11n.pt',
                      device='cpu',
                      conf_threshold=0.5)

# 加载分割工具
segment_tool = load_tool('SegmentAnything',
                         sam_model='sam_vit_h_4b8939.pth',
                         device='cuda')
```
### 2. 工具列表 (list_tools)
列出所有可用的工具
```python
from ag_mcxh.apis import list_tools

# 获取工具列表
tools = list_tools()
print(tools)  

# 获取工具列表及描述
tools_with_desc = list_tools(with_description=True)
for name, desc in tools_with_desc:
    print(f"{name}: {desc}")
```

### 3. 工具搜索 (search_tool)
根据查询语句搜索相关工具
```python
from ag_mcxh import search_tool

# 搜索相关工具
relevant_tools = search_tool("detect objects in image")
print(relevant_tools)
```

## 工具使用方法
### 图像输入/输出 (ImageIO)
所有工具都使用 ImageIO 类处理图像：
```python
from ag_mcxh.types import ImageIO

image = ImageIO('/path/to/image.jpg')
image_array = image.to_array()
pil_image = image.to_pil()
```
### YOLO检测工具
```python
from ag_mcxh.apis import load_tool
from ag_mcxh.types import ImageIO

yolo_tool = load_tool('YoloDetect',
                      model_path='/path/to/yolo11n.pt',
                      device='cpu',
                      conf_threshold=0.5,
                      iou_threshold=0.45,
                      image_size=640)

image = ImageIO('/path/to/image.jpg')

detection_results = yolo_tool.apply(image)
print(detection_results) 
```
### 分割工具
```python
from ag_mcxh.apis import load_tool
from ag_mcxh.types import ImageIO

segment_tool = load_tool('SegmentAnything',
                         sam_model='sam_vit_h_4b8939.pth',
                         device='cuda')

image = ImageIO('/path/to/image.jpg')
segmentation_result = segment_tool.apply(image)
```

## 使用案例
### 基本使用案例
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ag_mcxh.apis import list_tools, load_tool
from ag_mcxh.types import ImageIO

def main():
    print("Available tools:")
    tools = list_tools(with_description=True)
    for name, desc in tools:
        print(f"  - {name}: {desc}")
    
    if tools:
        yolo_tool = load_tool('YoloDetect', device='cpu')
        image = ImageIO('/path/to/image.jpg')
        results = yolo_tool.apply(image)
        print("Detection results:")
        print(results)

if __name__ == "__main__":
    main()
```
## 注意事项
1.使用前请确保已安装相应工具的依赖库
2.模型文件需要单独下载并放置在指定路径
3.GPU加速需要安装CUDA并配置相应的深度学习框架
4.图像路径需要是可访问的本地文件路径