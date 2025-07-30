from typing import Dict, Type, Any
import importlib
from .base import BaseTool

# 工具注册表
TOOLS_REGISTRY: Dict[str, Type[BaseTool]] = {}

def register_tool(name: str = None):
    """工具注册装饰器"""
    def decorator(cls: Type[BaseTool]):
        tool_name = name or cls.__name__
        TOOLS_REGISTRY[tool_name] = cls
        return cls
    return decorator

def get_tool_cls(name: str) -> Type[BaseTool]:
    """获取工具类"""
    if name not in TOOLS_REGISTRY:
        raise ValueError(f"工具 {name} 未注册")
    return TOOLS_REGISTRY[name]

def list_tools() -> list:
    """列出所有已注册的工具"""
    return list(TOOLS_REGISTRY.keys())

def load_tool(name: str, **kwargs) -> BaseTool:
    """加载工具实例"""
    tool_cls = get_tool_cls(name)
    return tool_cls(**kwargs)

def auto_register_tools():
    """自动注册所有工具模块"""
    # 注册基础工具
    try:
        from .detection import YoloDetect
    except ImportError:
        pass
        
    try:
        from .segmentation import SegmentAnything, SegmentObject
    except ImportError:
        pass