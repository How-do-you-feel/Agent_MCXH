from typing import List, Tuple, Optional, Dict, Any
from ..tools.registry import load_tool, list_tools as list_registered_tools

def load_tool(tool_type: str, **kwargs) -> Any:
    return load_tool(tool_type, **kwargs)

def list_tools(with_description: bool = False) -> List[str] or List[Tuple[str, str]]:
    if with_description:
        return [(name, f"{name} 工具") for name in list_registered_tools()]
    else:
        return list_registered_tools()

def search_tool(query: str, topk: int = 5) -> List[str]:

    all_tools = list_registered_tools()

    matched_tools = []
    for tool in all_tools:
        if query.lower() in tool.lower():
            matched_tools.append(tool)
    
    if not matched_tools:
        matched_tools = all_tools[:topk]
    
    return matched_tools[:topk]