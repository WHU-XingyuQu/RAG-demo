# llm_only_qwen.py — 只使用大模型通义千问，无向量检索
import os, sys
from langchain.chat_models import ChatOpenAI

# 中文输出防乱码
sys.stdout.reconfigure(encoding='utf-8')

# 设置代理（可选）
os.environ["http_proxy"]  = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"

# 设置 API 参数
os.environ["OPENAI_API_KEY"]  = os.getenv("SILICONFLOW_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

# 初始化 LLM（通义千问）
llm = ChatOpenAI(
    model_name="Qwen/Qwen3-8B",
    temperature=0.7
)

chat_history = []
print("【欢迎使用通义千问 LLM 生成实验助手（无向量检索）】")

# 交互式问答
while True:
    query = input("请输入问题（输入 'exit' 退出）：")
    if query.lower() == 'exit':
        print("感谢使用，再见！")
        break

    # 将上下文历史打包成 prompt（可选）
    context_prompt = "\n".join([f"用户：{q}\n助手：{a}" for q, a in chat_history])
    full_prompt = context_prompt + f"\n用户：{query}\n助手："

    try:
        result = llm.invoke(full_prompt)
        answer = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        answer = f"生成回答时出错：{str(e)}"

    chat_history.append((query, answer))

    # 判断是否为代码类问题
    if any(k in query.lower() for k in ["写", "函数", "代码", "python", "如何实现"]):
        print("\n【代码回答】\n")
        print("```python\n" + answer.strip() + "\n```")
    else:
        print("\n【回答】\n" + answer + "\n")
