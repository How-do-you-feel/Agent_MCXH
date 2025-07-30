import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from ag_mcxh.tools.base import BaseTool
from ag_mcxh.meta import ToolMeta, Parameter
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class ChineseKnowledgeTool(BaseTool):
    def __init__(self):
        toolmeta = ToolMeta(
            name="ChineseKnowledge",
            description="A tool for answering Chinese language and literature general knowledge questions",
            inputs=(
                Parameter(name="question", type="str", description="The Chinese language or literature question to answer"),
            ),
            outputs=(
                Parameter(name="answer", type="str", description="The answer to the question"),
            )
        )
        super().__init__(toolmeta)

    def apply(self, question):
        return question

chinese_tool = ChineseKnowledgeTool()

tool = Tool(
    name="ChineseKnowledge",
    func=chinese_tool.apply,
    description="Useful for answering Chinese language and literature general knowledge questions"
)

model_path = "/home/ps/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)
prompt = PromptTemplate.from_template("""
请作为一个语文知识专家，回答以下问题。请确保回答准确、专业且易于理解。

可用的工具：
{tools}
工具名称：
{tool_names}

问题：{input}
{agent_scratchpad}

请按照以下步骤回答：
1. 首先思考是否需要使用工具
2. 如果需要，选择合适的工具并执行
3. 基于工具执行结果，提供最终答案
4. 使用简洁明了的语言回答

回答：
""")

tools = [tool]
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="force"
)

def run_agent(question):
    response = agent_executor.invoke({
        "input": question
    })
    return response['output']

if __name__ == "__main__":
    question = "《红楼梦》的作者是谁？"
    print(run_agent(question))
