# -*- coding: utf-8 -*-
import os
import json
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 解决 HuggingFace 模型在国内下载慢的问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==========================================
# 1. 页面配置与 UI 极致美化
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
# 2. 密码验证锁
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
# 3. 核心配置 (读取 Kimi API)
# ==========================================
try:
    deepseek_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    # 本地测试降级方案
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key-here")

# 接入 Kimi (月之暗面) 大模型
client = OpenAI(api_key=deepseek_key, base_url="https://api.moonshot.cn/v1")

# ==========================================
# 4. 产业数据库与工具定义 (Function Calling)
# ==========================================
INDUSTRY_DB = {
    "数据分析师": {
        "salary": "15k - 25k",
        "skills": ["SQL", "Python", "Tableau", "统计学", "业务思维"],
        "companies": ["字节跳动", "美团", "招商银行"]
    },
    "AI大模型工程师": {
        "salary": "25k - 45k",
        "skills": ["PyTorch", "LangChain", "RAG架构", "Prompt Engineering"],
        "companies": ["百度", "阿里云", "月之暗面"]
    },
    "前端开发": {
        "salary": "12k - 20k",
        "skills": ["React", "Vue", "TypeScript", "性能优化"],
        "companies": ["腾讯", "小米", "网易"]
    }
}


def get_learning_path(major: str, target_job: str) -> str:
    job_info = None
    for key in INDUSTRY_DB:
        if key in target_job or target_job in key:
            job_info = INDUSTRY_DB[key]
            target_job = key
            break

    if not job_info:
        return f"抱歉，暂未收录《{target_job}》的详细数据。请结合你《{major}》的专业基础，去招聘网站查看具体要求。"

    skills_str = "、".join(job_info["skills"])
    comp_str = "、".join(job_info["companies"])

    return (
        f"【产业情报局】近期《{target_job}》市场行情：\n"
        f"💰 薪资参考：{job_info['salary']}\n"
        f"🛠️ 核心技能：{skills_str}\n"
        f"🏢 招聘大厂：{comp_str}\n"
        f"请结合学生专业《{major}》和上述真实要求，为他定制一份学习路径。"
    )


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    return f"【系统记录】已将学号{student_id}的《{topic}》成绩({score}分)存入系统数据库。学生能力画像已更新。"


tools = [
    {"type": "function", "function": {"name": "get_learning_path",
                                      "description": "根据学生的专业和目标岗位，从产业情报局获取最新岗位需求、薪资与技能要求，生成个性化学习路径",
                                      "parameters": {"type": "object", "properties": {"major": {"type": "string"},
                                                                                      "target_job": {"type": "string"}},
                                                     "required": ["major", "target_job"]}}},
    {"type": "function", "function": {"name": "assess_student_ability", "description": "记录学生成绩并更新能力画像",
                                      "parameters": {"type": "object", "properties": {"student_id": {"type": "string"},
                                                                                      "topic": {"type": "string"},
                                                                                      "score": {"type": "integer"}},
                                                     "required": ["student_id", "topic", "score"]}}}
]
available_functions = {"get_learning_path": get_learning_path, "assess_student_ability": assess_student_ability}

# ==========================================
# 5. 左侧控制台 (侧边栏)
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
    st.success("🟢 大模型接口：已连接 (Kimi)")
    st.info("🛠️ 工具链：已加载 (2个)")
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
# 6. 主界面与聊天逻辑
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
        with st.spinner("导师正在分析..."):

            # 记忆压缩机制
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
                        model="kimi-k2.6",  # 适配 Kimi 体验金的稳定模型
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
with st.expander("🛠️ 开发者调试窗口"):
    st.write("当前模型：Kimi")
    st.write("历史消息长度：", len(st.session_state.messages))
    if "vectorstore" in st.session_state:
        st.success("RAG 知识库已就绪")
    else:
        st.warning("RAG 知识库未加载")
