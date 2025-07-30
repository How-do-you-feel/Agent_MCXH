from .apis import load_tool, list_tools
from .tool_finder import search_tool

# 直接从agent.vision_agent导入VisionAgent
try:
    from .agent.vision_agent import VisionAgent
except ImportError as e:
    print(f"无法从agent.vision_agent导入VisionAgent: {e}")
    VisionAgent = None

try:
    from .version import __version__
except ImportError:
    __version__ = 'unknown'

# 只有在VisionAgent成功导入时才将其包含在__all__中
if VisionAgent is not None:
    __all__ = ['load_tool', 'list_tools', 'search_tool', 'VisionAgent', '__version__']
else:
    __all__ = ['load_tool', 'list_tools', 'search_tool', '__version__']