from typing import Dict, Any, Optional
# 修复导入路径
from ..apis.tool import load_tool
from ..tool_finder import search_tool
from .tool_selector import ToolSelector
from .vllm_client import VLLMHTTPClient, VLLMAsyncClient

class VisionAgent:
    """视觉智能体，可以根据自然语言提示自动选择和调用视觉工具"""
    
    def __init__(self, model_path: str = "/home/ps/Qwen2.5-3B", 
                 host: str = "127.0.0.1", port: int = 8001):
        # 初始化vLLM服务器管理器
        self.vllm_manager = VLLMAsyncClient(model_path, host, port)
        self.vllm_manager.start_server()
        self.vllm_client = self.vllm_manager.get_client()
        self.tool_selector = ToolSelector()
    
    def __del__(self):
        """析构函数，确保服务器被正确关闭"""
        if hasattr(self, 'vllm_manager'):
            self.vllm_manager.stop_server()
    
    def _execute_tool(self, tool_name: str, image_path: str, **kwargs) -> str:
        """执行指定的工具"""
        try:
            tool = load_tool(tool_name, **kwargs)
            from ..types import ImageIO
            image = ImageIO(image_path)
            result = tool.apply(image)
            return result
        except Exception as e:
            return f"执行工具时出错: {str(e)}"
    
    def process_with_vllm(self, prompt: str, image_path: str, **tool_kwargs) -> str:
        """使用vLLM模型根据提示和图像处理任务
        
        Args:
            prompt (str): 用户的自然语言请求
            image_path (str): 图像文件路径
            **tool_kwargs: 传递给工具的额外参数
            
        Returns:
            str: 工具执行结果
        """
        # 选择工具
        tool_name = self.tool_selector.select_tool_with_vllm(self.vllm_client, prompt, image_path)
        
        # 验证工具是否存在
        available_tools = self.tool_selector.get_available_tools()
        if tool_name not in available_tools:
            # 如果选择的工具不存在，使用搜索功能查找最接近的工具
            try:
                search_results = search_tool(prompt, topk=1)
                if search_results:
                    tool_name = search_results[0]
                else:
                    tool_name = available_tools[0]  # 默认使用第一个工具
            except:
                tool_name = available_tools[0]  # 默认使用第一个工具
        
        # 执行工具
        result = self._execute_tool(tool_name, image_path, **tool_kwargs)
        return result
    
    def direct_tool_call(self, tool_name: str, image_path: str, **tool_kwargs) -> str:
        """直接调用指定的工具
        
        Args:
            tool_name (str): 工具名称
            image_path (str): 图像文件路径
            **tool_kwargs: 传递给工具的额外参数
            
        Returns:
            str: 工具执行结果
        """
        return self._execute_tool(tool_name, image_path, **tool_kwargs)