import importlib
import inspect
from typing import Optional, List, Dict, Any

from ..tools import BaseTool

NAMES2TOOLS = {}

def extract_all_tools(module):
    if isinstance(module, str):
        module = importlib.import_module(module)

    tools = {}
    for k, v in module.__dict__.items():
        if isinstance(v, type) and issubclass(v, BaseTool) and v is not BaseTool:
            tools[k] = v
    return tools

def register_all_tools(module):
    NAMES2TOOLS.update(extract_all_tools(module))

def list_tools(with_description=False):
    if with_description:
        return [(name, cls.__doc__ or '') for name, cls in NAMES2TOOLS.items()]
    else:
        return list(NAMES2TOOLS.keys())

def load_tool(tool_type: str, **kwargs) -> BaseTool:
    if tool_type not in NAMES2TOOLS:
        raise ValueError(f'{tool_type} is not supported. Available tools: {list(NAMES2TOOLS.keys())}')

    constructor = NAMES2TOOLS[tool_type]
    tool_obj = constructor(**kwargs)
    return tool_obj

# 自动注册工具模块
try:
    from ..tools import base
    register_all_tools(base)
    
    from ..tools.segmentation import segment_anything
    register_all_tools(segment_anything)
    
    from ..tools.Yolo_Detect import yolo_detect
    register_all_tools(yolo_detect)
    
except ImportError as e:
    pass