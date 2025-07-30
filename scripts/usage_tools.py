from ag_mcxh.tools.base import BaseTool
from ag_mcxh.tools.registry import register_tool

@register_tool("MyNewTool")
class MyNewTool(BaseTool):
    @property
    def default_desc(self) -> str:
        return "This is my new tool"
    
    def apply(self, input_data):
        # Implement tool functionality
        return "Processing result"