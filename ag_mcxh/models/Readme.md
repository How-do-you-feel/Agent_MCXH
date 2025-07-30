## 注册新模型

本文档介绍了如何在MCXH Agent框架中注册新模型。

### 模型注册机制

模型通过`models/registry.py`中的注册机制进行管理。有两种注册模型的方式：

1. 使用[@register_model](file:///home/ps/MCXH/Agent_MCXH/ag_mcxh/models/registry.py#L6-L14)装饰器直接注册模型类
2. 使用[register_model_loader](file:///home/ps/MCXH/Agent_MCXH/ag_mcxh/models/registry.py#L16-L18)函数注册模型加载器（适用于需要延迟加载的模型）

### 注册新模型的方法

#### 方法一：使用装饰器注册（推荐用于简单模型）

创建新的模型文件（例如`models/my_model.py`）：

```python
from .registry import register_model

@register_model("MyModelName")
class MyModel:
    """模型描述"""
    
    def __init__(self, model_path: str = "default_path.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._model = None
    
    def load(self):
        """加载模型"""
        if self._model is None:
            # 实际加载模型的代码
            # from some_library import SomeModel
            # self._model = SomeModel(self.model_path)
            # self._model.to(self.device)
            pass
        return self._model
    
    def predict(self, input_data, **kwargs):
        """执行预测"""
        model = self.load()
        # 处理输入数据
        # results = model(input_data, **kwargs)
        # 返回结果
        return "prediction results"
```
#### 方法二：使用加载器函数注册（推荐用于复杂依赖）
```python
from .registry import register_model_loader

def _load_my_model():
    """延迟加载模型"""
    try:
        from some_library import SomeModel
        return SomeModel
    except ImportError:
        # 提供一个模拟类以避免导入错误
        class MockModel:
            def __init__(self, *args, **kwargs):
                pass
            def to(self, device):
                pass
            def __call__(self, *args, **kwargs):
                return []
        return MockModel

# 注册模型加载器
register_model_loader("MyModel", _load_my_model)
```
### 现有模型示例
参考`models/yolo.py`和`models/sam.py`中的实现了解更详细的示例。
### 使用已注册的模型
```python
from ag_mcxh.models import load_model

model = load_model("MyModelName", model_path="path/to/model.pt", device="cuda")
results = model.predict("path/to/input/data")
```
