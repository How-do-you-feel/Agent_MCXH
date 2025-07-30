import argparse
import sys
import os

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
            image_path = "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
            prompt_text = inputs.strip()
        return image_path, prompt_text

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
            temperature=0.7,
            max_tokens=512,
        )
        print(f"成功连接到vLLM服务: http://localhost:{args.model_port}/v1")
    except Exception as e:
        print(f"无法连接到vLLM服务: {e}")
        print("使用ChatOpenAI作为备用")
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key="your-api-key-here")

    # 使用hub提供的提示模板，它包含了所需的变量
    try:
        agent_prompt = hub.pull('hwchase17/structured-chat-agent')
    except Exception as e:
        # 如果无法从hub获取提示模板，则使用自定义模板
        print(f"无法从hub获取提示模板: {e}，使用自定义模板")
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个视觉智能体助手。你可以：
1. 回答一般性问题
2. 处理与图像相关的任务，如目标检测、图像分割、图像问答等

当用户询问与图像相关的问题时，你应该使用相应的工具。
对于一般性问题，你可以直接回答。

工具列表:
{tools}

工具名称:
{tool_names}"""),
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
        handle_parsing_errors=True
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
            user = prompt_toolkit_prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('输入错误')
            continue
        except KeyboardInterrupt:
            print("\n再见!")
            break
            
        if user.lower() == 'exit':
            print("再见!")
            break
            
        try:
            res = agent_executor.invoke({"input": user})
            print(f'\033[91mAgent\033[0m: {res["output"]}')
        except Exception as e:
            # 如果Agent执行失败，尝试直接使用LLM回答
            try:
                direct_response = llm.invoke(user)
                print(f'\033[91mAgent\033[0m: {direct_response}')
            except Exception as direct_e:
                print(f'\033[91mError\033[0m: {e}')
                print(f'直接调用LLM也失败了: {direct_e}')

if __name__ == "__main__":
    main()