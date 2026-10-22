# -*- coding: utf-8 -*-
import os

# 解决 HuggingFace 模型下载慢的问题，使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import json
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 页面基础配置与安全验证（比赛要求：账号密码）
# ==========================================
st.set_page_config(page_title="全场景教育产业智能体", page_icon="🎓", layout="wide")

# --- 简单的密码锁 ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.title("🔐 专家评审专用通道")
    pwd = st.text_input("请输入评审密码：", type="password")
    if st.button("进入系统"):
        if pwd == "judge2026":  # 👈 【修改这里】设置你的比赛评审密码
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("密码错误！")
    st.stop()  # 密码不对，停止运行后续代码

# ==========================================
# 2. 核心客户端与大模型初始化
# ==========================================
# ⚠️ 【修改这里】填入你真实的 DeepSeek API Key
import os
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY","sk-你的本地Key"),
    base_url="https://api.deepseek.com"
)


# 初始化本地向量模型 (第一次运行会下载约100MB)
@st.cache_resource(show_spinner="正在加载本地知识引擎...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


embeddings = get_embeddings()


# ==========================================
# 3. 定义业务工具 (Function Calling)
# ==========================================
def get_learning_path(major: str, target_job: str) -> str:
    return f"【系统数据】根据《{target_job}》的产业需求，该专业需要补充学习：1. 大模型API调用；2. LangChain框架。建议周期3周。"


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    return f"【系统数据】评估完成：学号{student_id}，{topic}得分为{score}。{'已掌握' if score >= 80 else '基础薄弱，建议复习'}。"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_learning_path",
            "description": "根据学生的专业和目标岗位，对比产业需求，生成个性化学习路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "major": {"type": "string", "description": "学生专业"},
                    "target_job": {"type": "string", "description": "目标岗位"}
                },
                "required": ["major", "target_job"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_student_ability",
            "description": "评估学生的学习成果，并记录成绩",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string", "description": "学号"},
                    "topic": {"type": "string", "description": "知识点"},
                    "score": {"type": "integer", "description": "得分0-100"}
                },
                "required": ["student_id", "topic", "score"]
            }
        }
    }
]
available_functions = {"get_learning_path": get_learning_path, "assess_student_ability": assess_student_ability}

# ==========================================
# 4. 侧边栏：RAG 知识库上传
# ==========================================
with st.sidebar:
    st.header("📚 行业教材知识库")
    st.info("请上传教学大纲或产业报告PDF，智能体将基于此文档进行回答。")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")

    if st.button("构建/更新知识库") and uploaded_file is not None:
        with st.spinner("正在解析教材并建立向量索引..."):
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # 存入 Session State 的向量库中
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            os.remove(file_path)
            st.success("✅ 知识库构建完成！")

# ==========================================
# 5. 主界面：聊天循环
# ==========================================
st.title("🎓 全场景 AI 智慧导师")
st.caption("基于大模型、RAG 与多工具调用的个性化学习与产业对接系统")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个全能的教育智能体。请根据用户输入，调用工具或结合教材片段回答问题。"}]

# 渲染历史记录
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("例如：我想做大模型工程师怎么学？/ 这篇PDF讲了什么？"):
    # 记录用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 开始思考
    with st.chat_message("assistant"):
        with st.spinner("导师正在分析需求并检索资料..."):

            # --- RAG 检索增强逻辑 ---
            rag_context = ""
            if "vectorstore" in st.session_state:
                # 去向量库搜索最相关的 3 个片段
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(prompt)
                if docs:
                    rag_context = "\n\n".join([doc.page_content for doc in docs])

            # 动态构建 System Prompt (融合教材内容)
            current_messages = st.session_state.messages.copy()
            if rag_context:
                current_messages[0][
                    "content"] += f"\n\n【参考教材内容】：\n{rag_context}\n请优先根据上述教材内容回答，如果教材没写，再使用你的常识或调用工具。"

            # --- Agent 工具调用循环 ---
            while True:
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=current_messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                    msg = response.choices[0].message

                    # 如果不需要调用工具，直接输出最终结果
                    if not msg.tool_calls:
                        st.markdown(msg.content)
                        st.session_state.messages.append({"role": "assistant", "content": msg.content})
                        break

                    # 如果需要调用工具
                    current_messages.append(msg)
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        st.toast(f"⚙️ 正在调用系统工具：{func_name}...")

                        # 执行本地函数
                        func_result = available_functions[func_name](**func_args)

                        # 将结果回传给大模型
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(func_result)
                        })
                except Exception as e:
                    st.error(f"💥 运行报错：{e}")
                    break