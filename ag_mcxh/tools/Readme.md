## 添加新工具

本文档介绍了如何在MCXH Agent框架中添加新工具。

### 工具注册机制

工具通过`tools/registry.py`中的注册机制进行管理。工具需要继承[BaseTool](base.py#L3-L31)类并使用[@register_tool](registry.py#L7-L13)装饰器注册。

### 创建新工具的方法
创建新的工具文件（例如`tools/my_tool.py`）：

```python
from .base import BaseTool
from .registry import register_tool

@register_tool("MyToolName")
class MyTool(BaseTool):
    """工具描述"""
    
    @property
    def default_desc(self) -> str:
        return "工具的默认描述"
    
    def __init__(self, 
                 param1: str = "default_value",
                 param2: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        self._resource = None
        
    def setup(self):
        # 初始化代码
        # self._resource = SomeResource()
        pass
    
    def apply(self, input_data):
        # 处理输入数据
        # result = self._resource.process(input_data)
        # 返回结果
        return "处理结果"
```
### 工具参数说明
- `toolmeta：工具元数据，可包含名称、描述等信息
- `setup()`：工具初始化方法，在第一次调用前执行
- `apply()`：工具主要功能实现，必须实现的抽象方法
- `call()`：使工具可调用，会自动调用setup()和apply()

#### 现有工具实例
参考`tools/detection.py`和`tools/segmentation.py`中的实现了解更详细的示例。
```python
from ag_mcxh.tools import load_tool

tool = load_tool("MyToolName", param1="value1", param2=20)
results = tool("input_data")
```

#### 自动注册
为了确保新添加的工具能被自动发现，需要在`tools/registry.py`的[auto_register_tools()](registry.py)函数中添加导入语句：
```python
def auto_register_tools():
    """自动注册所有工具模块"""
    # 注册现有工具
    try:
        from .detection import YoloDetect
    except ImportError:
        pass
        
    try:
        from .segmentation import SegmentAnything, SegmentObject
    except ImportError:
        pass
    
    try:
        from .my_tool import MyTool
    except ImportError:
        pass
```
