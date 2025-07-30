import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from prompt_toolkit import ANSI, prompt

# 导入视觉智能体
from ag_mcxh import VisionAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/ps/Qwen2.5-3B')
    parser.add_argument('--model-port', type=int, default=8001)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--vllm-port', type=int, default=8001)
    args = parser.parse_args()
    return args

class VisionAgentWrapper:
    """包装VisionAgent以适配LangChain"""
    
    def __init__(self, model_path, host, port):
        self.agent = VisionAgent(model_path=model_path, host=host, port=port)
        print("视觉智能体初始化成功")
    
    def process_with_vision_agent(self, inputs):
        """处理视觉任务"""
        prompt = inputs.get("prompt", "")
        image_path = inputs.get("image_path", "")
        
        if not prompt or not image_path:
            return "错误: 需要提供提示和图像路径"
        
        try:
            result = self.agent.process_with_vllm(prompt, image_path)
            return result
        except Exception as e:
            return f"处理图像时出错: {str(e)}"
    
    def direct_tool_call(self, inputs):
        """直接调用工具"""
        tool_name = inputs.get("tool_name", "")
        image_path = inputs.get("image_path", "")
        
        if not tool_name or not image_path:
            return "错误: 需要提供工具名称和图像路径"
        
        try:
            result = self.agent.direct_tool_call(tool_name, image_path, device="cpu")
            return result
        except Exception as e:
            return f"调用工具时出错: {str(e)}"

def format_vision_input(user_input):
    """解析用户输入，提取提示和图像路径"""
    # 简单解析，实际应用中可能需要更复杂的逻辑
    if "图像" in user_input or "图片" in user_input:
        # 这里可以添加更智能的解析逻辑
        return {
            "prompt": user_input,
            "image_path": "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"  # 默认图像路径
        }
    return {
        "prompt": user_input,
        "image_path": "/home/ps/MCXH/MingChaXinHao/ag_mcxh/pics/002.png"
    }

def main():
    args = parse_args()

    # 初始化视觉智能体
    try:
        vision_agent_wrapper = VisionAgentWrapper(
            model_path=args.model,
            host=args.host,
            port=args.vllm_port
        )
        print(f"成功初始化视觉智能体，使用模型: {args.model}")
    except Exception as e:
        print(f"初始化视觉智能体失败: {e}")
        return

    # 使用vLLM加载本地模型
    try:
        llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{args.model_port}/v1",
            model_name=args.model,
            temperature=0.7,
            max_tokens=512,
        )
        print(f"成功加载语言模型: {args.model}")
    except Exception as e:
        print(f"加载语言模型失败: {e}")
        print("使用默认的ChatOpenAI")
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # 创建处理链
    vision_chain = (
        RunnableLambda(format_vision_input)
        | RunnableLambda(vision_agent_wrapper.process_with_vision_agent)
    )
    
    # 创建最终的处理链
    template = """
    你是一个视觉智能体助手。用户会向你询问关于图像的问题。
    如果问题涉及图像分析，请使用视觉工具处理。
    如果是一般性问题，直接回答。

    用户问题: {question}
    视觉分析结果: {vision_result}

    请结合视觉分析结果回答用户问题:
    """

    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"question": RunnablePassthrough(), 
         "vision_result": vision_chain}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n视觉智能体LangChain版本已启动!")
    print("提示:")
    print("1. 你可以询问关于图像的问题")
    print("2. 智能体会自动分析图像并回答问题")
    print("3. 输入 'exit' 退出程序")
    print("-" * 50)

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        except KeyboardInterrupt:
            print("\n再见!")
            break
            
        if user.lower() == 'exit':
            print("再见!")
            break
            
        try:
            res = chain.invoke(user)
            print(f'\033[91mVisionAgent\033[0m: {res}')
        except Exception as e:
            print(f'\033[91mError\033[0m: {e}')

if __name__ == "__main__":
    main()