from typing import List, Tuple
from ..apis import list_tools

class ToolSelector:
    """工具选择器，根据提示选择合适的视觉工具"""
    
    def __init__(self):
        self.tools = list_tools(with_description=True)
    
    def build_tool_descriptions(self) -> str:
        """构建工具描述字符串"""
        tool_descs = []
        for name, desc in self.tools:
            tool_descs.append(f"- {name}: {desc}")
        return "\n".join(tool_descs)
    
    def select_tool_with_vllm(self, vllm_client, prompt: str, image_path: str) -> str:
        """使用vLLM模型根据用户提示和图像选择合适的工具"""
        tool_descriptions = self.build_tool_descriptions()
        
        # 构建系统提示
        system_prompt = f"""
你是一个视觉智能体助手，可以根据用户的请求选择最合适的视觉工具来处理图像。

可用的工具如下：
{tool_descriptions}

请根据用户的请求选择一个最合适的工具。只返回工具名称，不要包含其他内容。
例如：YoloDetect
        """.strip()
        
        # 构建完整的提示
        full_prompt = f"{system_prompt}\n\n用户请求: {prompt}"
        
        # 生成响应
        response = vllm_client.generate(full_prompt)
        return response.strip()
    
    def get_available_tools(self) -> List[str]:
        """获取所有可用工具的名称列表"""
        return [name for name, _ in self.tools]