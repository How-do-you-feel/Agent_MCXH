# 自定义工具开发指南

## 基础工具开发

### 1. 工具类继承与实现

所有自定义工具都需要继承 `BaseTool` 类。以语文知识工具为例：

```python
from ag_mcxh.tools.base import BaseTool
from ag_mcxh.meta import ToolMeta, Parameter

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
```
### 2.工具元数据配置
工具元数据通过`ToolMeta`类定义，包含：

- `name`: 工具名称
- `description`: 工具描述
- `inputs`: 输入参数定义
- `outputs`: 输出参数定义

每个参数使用`Parameter`类定义，包含：
- `name`: 参数名
- `type`: 参数类型
- `description`: 参数描述

### 3.工具实例化瘀使用
```python
# 初始化工具
chinese_tool = ChineseKnowledgeTool()

# 创建LangChain工具包装器
tool = Tool(
    name="ChineseKnowledge",
    func=chinese_tool.apply,
    description="Useful for answering Chinese language and literature general knowledge questions"
)
```
## 智能体集成
### 1.创建智能体
```python
# 创建智能体
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
```
### 2.提示模板配置

```python
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
```
### 3.智能体调用
```python
def run_agent(question):
    response = agent_executor.invoke({
        "input": question
    })
    return response['output']

# 测试
if __name__ == "__main__":
    question = "《红楼梦》的作者是谁？"
    print(run_agent(question))
```

- [案例完整代码在这](/ag_mcxh/examples/knoledge.py)
## 最佳实践
- 工具描述：提供清晰、准确的工具描述，帮助智能体理解工具功能
- 参数定义：使用类型注解明确定义输入输出类型
- 错误处理：在工具实现中添加适当的错误处理机制
- 性能优化：使用 setup 方法延迟加载重型模块
- 测试验证：确保工具在各种输入情况下都能正常工作
