from typing import Dict, Any, Optional
from .registry import load_tool, list_tools

def load_tool_by_name(name: str, **kwargs) -> Any:
    """根据名称加载工具"""
    return load_tool(name, **kwargs)

def get_available_tools() -> list:
    """获取可用工具列表"""
    return list_tools()

def create_tool_instance(name: str, **kwargs) -> Any:
    """创建工具实例"""
    return load_tool(name, **kwargs)