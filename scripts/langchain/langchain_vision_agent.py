import argparse
import sys
import os
import json

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
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8,
                        help='GPU memory utilization for vLLM (0.0-1.0)')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Maximum model length')
    return parser.parse_args()

def load_prompt_template():
    """加载提示词模板"""
    # 主要提示词文件路径
    template_path = "/home/ps/MCXH/Agent_MCXH/ag_mcxh/prompt/prompt.json"
    # 默认提示词文件路径
    default_template_path = "/home/ps/MCXH/Agent_MCXH/ag_mcxh/prompt/default_prompt.json"
    
    try:
        # 尝试加载主要提示词文件
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        return template_data["system_prompt"]
    except Exception as e:
        print(f"无法加载提示词模板 {template_path}: {e}")
        try:
            # 尝试加载默认提示词文件
            with open(default_template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            return template_data["system_prompt"]
        except Exception as default_e:
            print(f"无法加载默认提示词模板 {default_template_path}: {default_e}")
            # 如果两个文件都无法加载，抛出异常
            raise Exception("无法加载任何提示词模板文件")

class VisionAgentTool:
    def __init__(self, model_path="/home/ps/Qwen2.5-3B", host="127.0.0.1", port=8001,
                 gpu_memory_utilization=0.8, max_model_len=None):
        # VisionAgent不支持gpu_memory_utilization和max_model_len参数，所以不传递它们
        self.agent = VisionAgent(model_path=model_path, host=host, port=port)
    
    def yolo_detect(self, inputs: str) -> str:
        try:
            image_path, prompt_text = self._parse_inputs(inputs)
            result = self.agent.direct_tool_call("YoloDetect", image_path, device="cpu")
            return str(result)
        except Exception as e:
            return f"YOLO检测失败: {str(e)}"
    
    def segment_anything(self, inputs: str) -> str:
        try:
            image_path, prompt_text = self._parse_inputs(inputs)
            result = self.agent.direct_tool_call("SegmentAnything", image_path, device="cpu")
            return str(result)
        except Exception as e:
            return f"图像分割失败: {str(e)}"
    
    def ask_about_image(self, inputs: str) -> str:
        try:
            image_path, prompt_text = self._parse_inputs(inputs)
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                return f"错误：找不到图像文件 {image_path}"
            result = self.agent.process_with_vllm(prompt_text, image_path)
            return str(result)
        except Exception as e:
            return f"处理问题失败: {str(e)}"
    
    def _parse_inputs(self, inputs: str) -> tuple:
        parts = inputs.split("|")
        if len(parts) >= 2:
            image_path = parts[0].strip()
            prompt_text = "|".join(parts[1:]).strip()
        else:
            # 如果没有提供图像路径，使用默认路径
            image_path = "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
            prompt_text = inputs.strip()
        return image_path, prompt_text

def is_image_related_query(query):
    """判断查询是否与图像相关"""
    image_keywords = ["图像", "图片", "照片", "image", "picture", "photo", "检测", "分割", "识别", "yolo", "目标", "有什么"]
    return any(keyword in query.lower() for keyword in image_keywords)

def is_general_question(query):
    """判断是否为一般性问题"""
    general_keywords = ["你是谁", "你叫什么", "你是?", "who are you", "what is your name"]
    return any(keyword in query.lower() for keyword in general_keywords)

def get_direct_answer(query):
    """为常见问题提供直接答案"""
    query_lower = query.lower()
    if "你是谁" in query_lower or "你叫什么" in query_lower or "who are you" in query_lower:
        return "我是视觉智能体助手，可以帮您处理图像相关的任务，如目标检测、图像分割等。我也可以回答一般性问题。"
    return None

def main():
    args = parse_args()

    # 尝试初始化VisionAgentTool，如果失败则使用基本工具
    try:
        vision_tool_wrapper = VisionAgentTool(
            model_path=args.model, 
            port=args.model_port
        )
        tools_available = True
    except Exception as e:
        print(f"警告: VisionAgent初始化失败: {e}")
        tools_available = False

    if tools_available:
        tools = [
            Tool(
                name="YOLO_Detection",
                func=vision_tool_wrapper.yolo_detect,
                description="使用YOLO模型检测图像中的对象。输入格式: '图像路径|检测图像中的所有对象'"
            ),
            Tool(
                name="Segment_Anything",
                func=vision_tool_wrapper.segment_anything,
                description="使用Segment Anything模型分割图像。输入格式: '图像路径|分割图像中的所有对象'"
            ),
            Tool(
                name="Ask_About_Image",
                func=vision_tool_wrapper.ask_about_image,
                description="询问关于图像的问题。输入格式: '图像路径|你的问题'"
            )
        ]
    else:
        # 创建基本工具，不依赖VisionAgent
        def dummy_tool(inputs: str) -> str:
            return "视觉工具当前不可用，请检查模型服务是否正常运行。"
        
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
            temperature=0.1,  # 降低温度
            max_tokens=300,   # 限制最大输出长度
        )
        print(f"成功连接到vLLM服务: http://localhost:{args.model_port}/v1")
    except Exception as e:
        print(f"无法连接到vLLM服务: {e}")
        print("使用ChatOpenAI作为备用")
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="your-api-key-here")

    # 加载提示词模板
    try:
        system_prompt = load_prompt_template()
    except Exception as e:
        print(f"加载提示词模板失败: {e}")
        return

    # 使用hub提供的提示模板，它包含了所需的变量
    try:
        agent_prompt = hub.pull('hwchase17/structured-chat-agent')
    except Exception as e:
        # 如果无法从hub获取提示模板，则使用自定义模板
        print(f"无法从hub获取提示模板: {e}，使用自定义模板")
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
        max_iterations=5,  # 限制最大迭代次数
        # 移除不支持的early_stopping_method参数
        return_intermediate_steps=True  # 返回中间步骤
    )

    print(f"视觉智能体LangChain版本已启动! 使用模型: {args.model}")
    print("提示:")
    print("1. 你可以询问关于图像的问题")
    print("2. 智能体会自动选择合适的视觉工具")
    print("3. 你可以指定使用特定工具，如 '使用YOLO检测图像'")
    print("4. 输入 'exit' 退出程序")
    print("-" * 50)

    conversation_count = 0
    while True:
        try:
            user_input = prompt_toolkit_prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('输入错误')
            continue
        except KeyboardInterrupt:
            print("\n再见!")
            break
            
        if user_input.lower() == 'exit':
            print("再见!")
            break
            
        conversation_count += 1
        print(f"[对话轮次: {conversation_count}]")
        
        # 检查是否为常见问题并提供直接答案
        direct_answer = get_direct_answer(user_input)
        if direct_answer:
            print(f'\033[91mAgent\033[0m: {direct_answer}')
            continue
            
        # 检查是否是图像相关问题
        if is_image_related_query(user_input):
            # 对于图像相关问题，使用Agent处理
            try:
                res = agent_executor.invoke({"input": user_input})
                # 限制输出长度，防止重复
                output = res["output"]
                if len(output) > 800:
                    output = output[:800] + "..."
                print(f'\033[91mAgent\033[0m: {output}')
                continue
            except Exception as e:
                print(f'\033[91mError\033[0m: {e}')
                # 如果Agent执行失败，尝试直接使用视觉工具
                try:
                    # 尝试直接调用Ask_About_Image工具
                    image_path = "/home/ps/MCXH/Agent_MCXH/pics/002.png"  # 默认图像路径
                    # 如果用户输入中包含路径，则使用用户提供的路径
                    if "/home/ps/" in user_input and ".png" in user_input:
                        # 简单提取路径
                        import re
                        match = re.search(r'/home/ps/[^ ]*\.png', user_input)
                        if match:
                            image_path = match.group(0)
                    
                    result = vision_tool_wrapper.ask_about_image(f"{image_path}|{user_input}")
                    print(f'\033[91mAgent\033[0m: {result}')
                    continue
                except Exception as direct_e:
                    print(f'直接调用视觉工具也失败了: {direct_e}')
        
        # 对于非图像相关问题，直接使用LLM回答
        try:
            direct_response = llm.invoke(user_input)
            # 限制输出长度，防止重复
            if len(direct_response) > 500:
                direct_response = direct_response[:500] + "..."
            print(f'\033[91mAgent\033[0m: {direct_response}')
            continue
        except Exception as direct_e:
            print(f'直接调用LLM失败: {direct_e}')

if __name__ == "__main__":
    main()