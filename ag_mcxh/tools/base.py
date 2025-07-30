from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, toolmeta: Optional[Dict[str, Any]] = None):
        self.toolmeta = toolmeta or {}
        self.name = self.toolmeta.get('name', self.__class__.__name__)
        self.description = self.toolmeta.get('description', self.default_desc)
        self._is_setup = False
    
    @property
    def default_desc(self) -> str:
        """默认工具描述"""
        return "这是一个基础工具"
    
    def setup(self):
        """工具初始化方法"""
        pass
    
    @abstractmethod
    def apply(self, *args, **kwargs):
        """工具应用方法"""
        pass
    
    def __call__(self, *args, **kwargs):
        """使工具可调用"""
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.apply(*args, **kwargs)