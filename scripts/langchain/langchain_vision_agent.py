import argparse
import sys
import os
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scripts_dir)
ag_mcxh_path = os.path.join(project_root, 'ag_mcxh')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if ag_mcxh_path not in sys.path:
    sys.path.insert(0, ag_mcxh_path)

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain import hub
from prompt_toolkit import ANSI, prompt as prompt_toolkit_prompt

from ag_mcxh import VisionAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/ps/Qwen2.5-3B')
    parser.add_argument('--model-port', type=int, default=8001)
    return parser.parse_args()

def load_prompt_template():
    template_path = "/home/ps/MCXH/Agent_MCXH/ag_mcxh/prompt/prompt.json"
    default_template_path = "/home/ps/MCXH/Agent_MCXH/ag_mcxh/prompt/default_prompt.json"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        return template_data["system_prompt"]
    except Exception:
        try:
            with open(default_template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            return template_data["system_prompt"]
        except Exception:
            raise Exception("无法加载任何提示词模板文件")

class VisionAgentTool:
    def __init__(self, model_path="/home/ps/Qwen2.5-3B", host="127.0.0.1", port=8001):
        self.agent = VisionAgent(model_path=model_path, host=host, port=port)
    
    def yolo_detect(self, inputs: str) -> str:
        try:
            image_path, prompt_text = self._parse_inputs(inputs)
            result = self.agent.direct_tool_call("YoloDetect", image_path, device="cpu")
            return str(result)
        except Exception as e:
            return f"YOLO检测失败: {str(e)}"
    
    def _parse_inputs(self, inputs: str) -> tuple:
        parts = inputs.split("|")
        if len(parts) >= 2:
            image_path = parts[0].strip()
            prompt_text = "|".join(parts[1:]).strip()
        else:
            image_path = "/home/ps/MCXH/Agent_MCXH/pics/002.png"
            prompt_text = inputs.strip()
        return image_path, prompt_text

def is_image_related_query(query):
    image_keywords = ["图像", "图片", "照片", "image", "picture", "photo", "检测", "分割", "识别", "yolo", "目标", "有什么"]
    return any(keyword in query.lower() for keyword in image_keywords)

def get_direct_answer(query):
    query_lower = query.lower()
    if "你是谁" in query_lower or "你叫什么" in query_lower or "who are you" in query_lower:
        return "我是视觉智能体助手，可以帮您处理图像相关的任务，如目标检测、图像分割等。我也可以回答一般性问题。"
    return None

def extract_image_path(query):
    match = re.search(r'/home/ps/[^ ]*?\.(png|jpg|jpeg|bmp|gif)', query)
    if match:
        return match.group(0)
    return "/home/ps/MCXH/Agent_MCXH/pics/002.png"

def main():
    args = parse_args()

    try:
        vision_tool_wrapper = VisionAgentTool(
            model_path=args.model, 
            port=args.model_port
        )
        tools_available = True
    except Exception:
        tools_available = False

    if tools_available:
        tools = [
            Tool(
                name="YOLO_Detection",
                func=vision_tool_wrapper.yolo_detect,
                description="使用YOLO模型检测图像中的对象。输入格式: '图像路径|检测图像中的所有对象'"
            )
        ]
    else:
        def dummy_tool(inputs: str) -> str:
            return "视觉工具当前不可用"
        
        tools = [
            Tool(
                name="Basic_Tool",
                func=dummy_tool,
                description="基础工具，当视觉工具不可用时使用"
            )
        ]

    try:
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{args.model_port}/v1",
            model_name=args.model,
            temperature=0.1,
            max_tokens=300,
        )
    except Exception:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="your-api-key-here")

    try:
        system_prompt = load_prompt_template()
    except Exception:
        return

    try:
        agent_prompt = hub.pull('hwchase17/structured-chat-agent')
    except Exception:
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

    agent = create_structured_chat_agent(
        llm,
        tools,
        prompt=agent_prompt
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="output"
    )
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )

    print(f"视觉智能体LangChain版本已启动! 使用模型: {args.model}")
    print("提示:")
    print("1. 你可以询问关于图像的问题")
    print("2. 智能体会自动选择合适的视觉工具")
    print("3. 你可以指定使用特定工具，如 '使用YOLO检测图像'")
    print("4. 输入 'exit' 退出程序")
    print("-" * 50)

    while True:
        try:
            user_input = prompt_toolkit_prompt(ANSI('\033[92mUser\033[0m: '))
        except:
            continue
            
        if user_input.lower() == 'exit':
            break
            
        direct_answer = get_direct_answer(user_input)
        if direct_answer:
            print(f'\033[91mAgent\033[0m: {direct_answer}')
            continue
            
        if is_image_related_query(user_input):
            try:
                res = agent_executor.invoke({"input": user_input})
                output = res["output"]
                if len(output) > 800:
                    output = output[:800] + "..."
                print(f'\033[91mAgent\033[0m: {output}')
                continue
            except Exception:
                pass
        
        try:
            direct_response = llm.invoke(user_input)
            if len(direct_response) > 500:
                direct_response = direct_response[:500] + "..."
            print(f'\033[91mAgent\033[0m: {direct_response}')
        except Exception:
            pass

if __name__ == "__main__":
    main()