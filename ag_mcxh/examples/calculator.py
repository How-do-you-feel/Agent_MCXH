import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ag_mcxh.tools.base import BaseTool
from ag_mcxh.meta import ToolMeta, Parameter


class CalculatorTool(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name="Calculator",
            description="A simple calculator that can perform basic arithmetic operations",
            inputs=(
                Parameter(name="x", type="float", description="First number"),
                Parameter(name="y", type="float", description="Second number"),
                Parameter(name="operation", type="str", description="Operation to perform (+, -, *, /)")
            ),
            outputs=(
                Parameter(name="result", type="float", description="Result of the calculation"),
            )
        )
        super().__init__(toolmeta)

    def apply(self, x, y, operation):
        operation_map = {
            '+': '加', '-': '减', '*': '乘', '/': '除',
            '加': '+', '减': '-', '乘': '*', '除': '/'
        }
        operation = operation_map.get(operation, operation)
        
        # 确保输入是float类型
        x = float(x)
        y = float(y)
        
        if operation == '+':
            return x + y
        elif operation == '-':
            return x - y
        elif operation == '*':
            return x * y
        elif operation == '/':
            if y == 0:
                raise ValueError("Cannot divide by zero")
            return x / y
        else:
            raise ValueError(f"Unsupported operation: {operation}")

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 初始化计算器工具
calculator_tool = CalculatorTool()


def parse_input(input_str):
    # 解析类似 "x=234 y=909 operation=*" 的输入
    parts = input_str.split()
    params = {}
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key.strip()] = value.strip()
    return params.get('x'), params.get('y'), params.get('operation')

tool = Tool(
    name="Calculator",
    func=lambda input_str: calculator_tool.apply(*parse_input(input_str)),
    description="Useful for performing mathematical calculations. Input should be in the format: x=number y=number operation=operator"
)


model_path = "/home/ps/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,  # 减少token数量
    temperature=0.01,    # 进一步降低温度
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)



llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate.from_template("""
请严格按照以下格式回答问题。你可以使用以下工具：

{tools}

工具名称: {tool_names}

格式要求：
Question: {input}
Thought: 思考过程
Action: Calculator
Action Input: x=数字 y=数字 operation=运算符
Observation: 工具执行的结果
Thought: 我现在知道最终答案了
Final Answer: 最终答案

注意：
1. 必须严格按照上述格式，每个标签单独一行
2. 不要在Observation之前生成Final Answer
3. 必须等待Observation之后才能生成Final Answer
4. Action必须是Calculator
5. Action Input必须严格按照"x=数字 y=数字 operation=运算符"的格式
6. 数字必须是具体的数值，不能是表达式
7. 运算符必须是 +, -, *, / 中的一个
8. 不要添加任何其他内容

开始！

Question: {input}
Thought:{agent_scratchpad}
""")






tools = [tool]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,  # 增加迭代次数
    early_stopping_method="force"
)






# 使用示例
def run_agent(question):
    response = agent_executor.invoke({
        "input": question
    })
    return response['output']

# 测试
if __name__ == "__main__":
    question = "234乘909等于多少"
    print(run_agent(question))

