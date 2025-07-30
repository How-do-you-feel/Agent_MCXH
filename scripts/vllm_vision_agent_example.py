#!/usr/bin/env python3
"""使用vLLM的Vision Agent使用示例"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ag_mcxh.agent.vllm_client import VLLMHTTPClient
from ag_mcxh.agent.tool_selector import ToolSelector
from ag_mcxh.apis import load_tool, search_tool

class SimpleVisionAgent:
    """简化的视觉智能体，连接到已有的vLLM服务器"""
    
    def __init__(self, vllm_base_url: str = "http://localhost:8000"):
        self.vllm_client = VLLMHTTPClient(vllm_base_url)
        self.tool_selector = ToolSelector()
    
    def _execute_tool(self, tool_name: str, image_path: str, **kwargs) -> str:
        """执行指定的工具"""
        try:
            tool = load_tool(tool_name, **kwargs)
            from ag_mcxh.types import ImageIO
            image = ImageIO(image_path)
            result = tool.apply(image)
            return result
        except Exception as e:
            return f"执行工具时出错: {str(e)}"
    
    def process_with_vllm(self, prompt: str, image_path: str, **tool_kwargs) -> str:
        """使用vLLM模型根据提示和图像处理任务"""
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
        """直接调用指定的工具"""
        return self._execute_tool(tool_name, image_path, **tool_kwargs)

def main():
    print("使用vLLM的Vision Agent 使用示例")
    print("=" * 40)
    
    # 连接到已有的vLLM服务器
    try:
        agent = SimpleVisionAgent("http://localhost:8001")
        print("✓ VisionAgent 连接到vLLM服务器成功")
    except Exception as e:
        print(f"✗ VisionAgent 连接失败: {e}")
        return
    
    # 示例图像路径和请求
    image_path = "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
    prompt = "请检测图像中的所有对象"
    
    if not os.path.exists(image_path):
        print(f"警告: 图像文件 {image_path} 不存在，请替换为实际的图像路径")
        return
    
    print(f"\n处理图像: {image_path}")
    print(f"用户请求: {prompt}")
    
    try:
        # 处理请求
        result = agent.process_with_vllm(prompt, image_path)
        print("\n处理结果:")
        print(result)
    except Exception as e:
        print(f"处理过程中出错: {e}")
    
    print("\n" + "-" * 40)
    print("直接调用YOLO工具示例:")
    
    try:
        # 直接调用YOLO工具
        result = agent.direct_tool_call("YoloDetect", image_path, device="cpu")
        print("YOLO检测结果:")
        print(result)
    except Exception as e:
        print(f"直接调用工具时出错: {e}")

if __name__ == "__main__":
    main()