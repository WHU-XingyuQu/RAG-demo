# rag_demo_qwen.py — 硅基流动通义千问 + 本地嵌入版
import os, sys
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# 中文输出防乱码
sys.stdout.reconfigure(encoding='utf-8')

# 设置代理（更换成您的代理）
os.environ["http_proxy"]  = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"

# 设置 API 参数
os.environ["OPENAI_API_KEY"]  = os.getenv("SILICONFLOW_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

# 1. 加载文档
loader   = TextLoader("data.txt", encoding="utf-8")
docs     = loader.load()
chunks   = CharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

# 2. 使用本地 HuggingFace 嵌入器（推荐）
emb      = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorDB = FAISS.from_documents(chunks, emb)

# 3. 使用硅基流动上的 Qwen 模型
llm = ChatOpenAI(
    model_name="Qwen/Qwen3-8B",
    temperature=0.3,
)
qa_chain=ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorDB.as_retriever(),
    return_source_documents=True,
)


chat_history = []
print("【欢迎使用硅基流动通义千问RAG实验智能助手】")

# 4. 交互
while True:
    query = input("有什么可以帮助您的（输入 'exit' 退出）：")
    if query.lower() == 'exit':
        print("感谢使用，再见！")
        break
    result = qa_chain.invoke({
    "question": query,
    "chat_history": chat_history
})

    if isinstance(result, dict):
        answer = result.get("answer", "")
    else:
        answer = result

    chat_history.append((query, answer))

    # 判断是否是代码类问题
    if any(k in query.lower() for k in ["写", "函数", "代码", "python", "如何实现"]):
        print("\n【代码回答】\n")
        print("```python\n" + answer.strip() + "\n```")
    else:
        print("\n【回答】\n" + answer + "\n")

