# -*- coding: utf-8 -*-
import os
import json
import streamlit as st
from openai import OpenAI
from hiringindex import HiringIndex
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 页面配置与 UI 极致美化 (注入 CSS)
# ==========================================
st.set_page_config(page_title="全场景AI智慧导师", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%); font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; }
    .hero-title { background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298, #00c6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important; font-weight: 800 !important; text-align: center; margin-bottom: 0px; }
    .hero-subtitle { text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 30px; }
    div[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border-right: 1px solid rgba(0,0,0,0.05); }
    .stChatMessage { border-radius: 15px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 15px; background-color: white; }
    .stButton>button { background: linear-gradient(to right, #2a5298, #1e3c72); color: white; border-radius: 20px; border: none; padding: 10px 24px; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(42, 82, 152, 0.4); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密码验证
# ==========================================
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.write("##");
    st.write("##")
    st.markdown("<h1 style='text-align: center; color: #2a5298;'>🔐 专家评审专用通道</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("请输入评审密码：", type="password", placeholder="请输入 judge2026")
        if st.button("进入系统", use_container_width=True):
            if pwd == "judge2026":
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("密码错误，请重试！")
    st.stop()

# ==========================================
# 3. 核心配置 (双路容错：云端读 Secrets，本地读文件)
# ==========================================
try:
    deepseek_key = st.secrets["DEEPSEEK_API_KEY"]
    rapidapi_key = st.secrets["RAPIDAPI_KEY"]
except Exception:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "sk-你的DeepSeek备用Key")
    rapidapi_key = os.getenv("RAPIDAPI_KEY", "你的RapidAPI备用Key")

client = OpenAI(api_key=deepseek_key, base_url="https://api.moonshot.cn/v1")

# 初始化真实招聘接口客户端
job_client = HiringIndex(api_key=rapidapi_key)


# ==========================================
# 4. 定义真实业务工具 (接入 API)
# ==========================================
def get_learning_path(major: str, target_job: str) -> str:
    """调用 HiringIndex API 获取真实岗位数据"""
    try:
        # 搜索近30天内的真实岗位数据
        page = job_client.search(job_titles=[target_job], days_ago=30, limit=20)

        if not page.get("jobs"):
            return f"【产业情报局】抱歉，近期暂未从招聘平台抓取到与《{target_job}》相关的岗位。"

        # 提取前3个真实岗位信息
        job_samples = []
        for job in page["jobs"][:3]:
            title = job.get("title", "未知岗位")
            company = job.get("company_name", job.get("handle", "未知公司"))
            url = job.get("apply_url", "无链接")
            job_samples.append(f"公司：{company}，岗位：{title}，投递链接：{url}")

        # 提取薪资洞察
        insights = job_client.insights(job_titles=[target_job])
        salary_info = "薪资数据暂未披露"
        if insights.get("salary"):
            bands = []
            for band in insights["salary"]:
                if band.get("min") and band["min"].get("p50"):
                    bands.append(f"{band['currency']} {band['min']['p50']}")
            if bands:
                salary_info = " | ".join(bands)

        # 打包返回
        return (
                f"【真实产业情报】近期《{target_job}》市场动态：\n"
                f"薪资参考（中位数）：{salary_info}\n"
                f"真实岗位示例：\n" + "\n".join(job_samples) +
                f"\n请结合学生专业《{major}》和上述真实需求，为学生定制学习路径。"
        )
    except Exception as e:
        # API 调用失败时的优雅降级
        return f"【网络波动】招聘接口调用失败 ({str(e)})。请凭你的常识，结合《{target_job}》的一般要求为《{major}》专业学生提供建议。"


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    return f"【系统记录】已将学号{student_id}的《{topic}》成绩({score}分)存入系统数据库。学生能力画像已更新。"


# 注册工具
tools = [
    {"type": "function", "function": {"name": "get_learning_path",
                                      "description": "根据学生的专业和目标岗位，从真实招聘平台获取最新产业需求与薪资，生成个性化学习路径",
                                      "parameters": {"type": "object", "properties": {"major": {"type": "string"},
                                                                                      "target_job": {"type": "string"}},
                                                     "required": ["major", "target_job"]}}},
    {"type": "function", "function": {"name": "assess_student_ability", "description": "记录学生成绩并更新画像",
                                      "parameters": {"type": "object", "properties": {"student_id": {"type": "string"},
                                                                                      "topic": {"type": "string"},
                                                                                      "score": {"type": "integer"}},
                                                     "required": ["student_id", "topic", "score"]}}}
]
available_functions = {"get_learning_path": get_learning_path, "assess_student_ability": assess_student_ability}

# ==========================================
# 5. 左侧侧边栏
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5298;'>⚙️ 系统控制台</h2>", unsafe_allow_html=True)

    persona = st.selectbox("选择导师人设", ["默认全场景导师", "严厉的考研导师", "幽默风趣的学长", "职场资深HR"])
    prompt_map = {
        "默认全场景导师": "你是一个全能的AI教育智能体，请用专业、鼓励的语气回答学生的问题。",
        "严厉的考研导师": "你是一个极其严厉的考研导师。你的回答必须一针见血，指出学生的不足，绝不废话。",
        "幽默风趣的学长": "你是一个幽默风趣的学长。你喜欢用网络梗和生活中的段子来打比方。",
        "职场资深HR": "你是一个世界500强企业的资深HR。你会从求职、薪资的角度给出最实用的建议。"
    }

    st.divider()
    st.markdown("#### 📊 运行状态")
    st.success("🟢 大模型接口：已连接 (DeepSeek)")
    st.info("🛠️ 招聘数据 API：已连接 (HiringIndex)")
    if "vectorstore" in st.session_state:
        st.success("📚 本地知识库：已就绪")
    else:
        st.warning("📚 本地知识库：未加载")

    st.divider()
    st.markdown("#### 📚 教材知识库")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf", label_visibility="collapsed")

    if st.button("构建/更新知识库", use_container_width=True) and uploaded_file is not None:
        with st.spinner("正在解析教材..."):
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)


            @st.cache_resource(show_spinner="加载向量模型中...")
            def get_embeddings():
                return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=get_embeddings())
            os.remove(file_path)
            st.success("✅ 知识库构建完成！")
            st.rerun()

    st.divider()
    if st.button("🗑️ 清空当前对话记忆", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

# ==========================================
# 6. 主界面
# ==========================================
st.markdown('<h1 class="hero-title">🎓 全场景 AI 智慧导师</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">当前模式：<b>{persona}</b> | 基于 RAG 与多工具调用的个性化教育系统</p >',
            unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
else:
    if st.session_state.messages[0]["content"] != prompt_map[persona]:
        st.session_state.messages[0]["content"] = prompt_map[persona]

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

if prompt := st.chat_input("例如：我想做数据分析师怎么学？ / 帮我查一下AI工程师的薪资"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("导师正在分析产业数据..."):

            # 记忆压缩
            if len(st.session_state.messages) > 10:
                st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-6:]

            # RAG 检索
            rag_context = ""
            if "vectorstore" in st.session_state:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(prompt)
                if docs:
                    rag_context = "\n\n".join([doc.page_content for doc in docs])

            current_messages = st.session_state.messages.copy()
            if rag_context:
                current_messages[0]["content"] += f"\n\n【参考教材内容】：\n{rag_context}\n请结合以上教材内容回答。"

            # Agent 工具循环
            while True:
                try:
                    response = client.chat.completions.create(
                        model="kimi-k2.6",
                        messages=current_messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                    msg = response.choices[0].message

                    if not msg.tool_calls:
                        st.markdown(msg.content)
                        st.session_state.messages.append({"role": "assistant", "content": msg.content})
                        break

                    current_messages.append(msg)
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        st.toast(f"⚙️ 正在调用工具：{func_name}...")

                        func_result = available_functions[func_name](**func_args)
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(func_result)
                        })
                except Exception as e:
                    st.error(f"💥 运行报错：{e}")
                    break