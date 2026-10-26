# -*- coding: utf-8 -*-
import os
import json
import time
import base64
import sqlite3
import datetime
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==========================================
# 1. 页面配置与 UI 美化
# ==========================================
st.set_page_config(page_title="启明 AI 导师", page_icon="🌟", layout="wide", initial_sidebar_state="expanded")

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
# 2. 数据库操作 (历史记录持久化)
# ==========================================
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    if isinstance(content, list):
        content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
    c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, role, str(content), datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_messages(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def get_all_sessions():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT session_id, MIN(content) FROM messages WHERE role='user' GROUP BY session_id ORDER BY MAX(timestamp) DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1][:15] + "..."} for row in rows]

def delete_session(session_id):
    conn = sqlite3.connect('chat_history.db')
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

init_db()
# 3. 密码锁
# ==========================================
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.write("##"); st.write("##")
    st.markdown("<h1 style='text-align: center; color: #2a5298;'>🌟 启明 AI · 专家评审通道</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
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
# 4. 核心客户端与工具
# ==========================================
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-你的Kimi真实Key")

client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

INDUSTRY_DB = {
    "数据分析师": {"salary": "15k - 25k", "skills": ["SQL", "Python", "Tableau", "统计学", "业务思维"], "companies": ["字节跳动", "美团", "招商银行"]},
    "AI大模型工程师": {"salary": "25k - 45k", "skills": ["PyTorch", "LangChain", "RAG架构", "Prompt Engineering"], "companies": ["百度", "阿里云", "月之暗面"]},
    "前端开发": {"salary": "12k - 20k", "skills": ["React", "Vue", "TypeScript", "性能优化"], "companies": ["腾讯", "小米", "网易"]}
}

def get_learning_path(major: str, target_job: str) -> str:
    job_info = None
    for key in INDUSTRY_DB:
        if key in target_job or target_job in key:
            job_info = INDUSTRY_DB[key]
            target_job = key
            break
    if not job_info:
        return f"抱歉，暂未收录《{target_job}》的详细数据。"
    return f"【产业情报】《{target_job}》薪资: {job_info['salary']}, 技能: {', '.join(job_info['skills'])}, 大厂: {', '.join(job_info['companies'])}。请结合《{major}》给出学习路径。"


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    return f"【系统记录】已将学号{student_id}的《{topic}》成绩({score}分)存入数据库。"


tools = [
    {"type": "function", "function": {"name": "get_learning_path", "description": "获取产业需求与薪资",
                                      "parameters": {"type": "object", "properties": {"major": {"type": "string"},
                                                                                      "target_job": {"type": "string"}},
                                                     "required": ["major", "target_job"]}}},
    {"type": "function", "function": {"name": "assess_student_ability", "description": "记录成绩",
                                      "parameters": {"type": "object", "properties": {"student_id": {"type": "string"},
                                                                                      "topic": {"type": "string"},
                                                                                      "score": {"type": "integer"}},
                                                     "required": ["student_id", "topic", "score"]}}}
]
available_functions = {"get_learning_path": get_learning_path, "assess_student_ability": assess_student_ability}

# ==========================================
# 5. 侧边栏 (历史记录、控制台、上传)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5298;'>⚙️ 系统控制台</h2>", unsafe_allow_html=True)

    # 5.1 初始化状态
    if "current_session_id" not in st.session_state:
        import uuid

        st.session_state.current_session_id = str(uuid.uuid4())[:8]

    # 5.2 角色定义与选择
    prompt_map = {
        "默认全场景导师": "你是一个全能的AI教育智能体，请用专业、鼓励的语气回答学生的问题。",
        "严厉的考研导师": "你是一个极其严厉的考研导师。你的回答必须一针见血。",
        "幽默风趣的学长": "你是一个幽默风趣的学长。你喜欢用网络梗来打比方。",
        "职场资深HR": "你是一个世界500强企业的资深HR。你会从求职、薪资角度给出建议。"
    }

    # 5.3 初始化对话状态 (防清空核心)
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": prompt_map["默认全场景导师"]}]

    persona = st.selectbox(
        "选择导师人设",
        ["默认全场景导师", "严厉的考研导师", "幽默风趣的学长", "职场资深HR"],
        key="persona_key"
    )

    # 只有当角色真的切换时，才更新 System Prompt
    if st.session_state.messages[0]["content"] != prompt_map[persona]:
        st.session_state.messages[0]["content"] = prompt_map[persona]

    # 5.4 PDF 知识库
    st.divider()
    st.markdown("#### 📚 教材知识库")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf", label_visibility="collapsed")
    if st.button("构建/更新知识库", use_container_width=True) and uploaded_file is not None:
        with st.spinner("正在解析教材..."):
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            docs = PyPDFLoader(file_path).load()
            splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)


            @st.cache_resource(show_spinner="加载向量模型中...")
            def get_embeddings():
                return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=get_embeddings())
            os.remove(file_path)
            st.success("✅ 知识库构建完成！")
            st.rerun()

    # 5.5 多模态图片上传
    st.divider()
    st.markdown("#### 🖼️ 多模态识图")
    uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True)

    # 5.6 历史记录管理
    st.divider()
    st.markdown("#### 🗂️ 历史记录")
    if st.button("➕ 开启全新对话", use_container_width=True):
        import uuid

        st.session_state.current_session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

    if st.button("👑 清空所有历史记录", use_container_width=True):
        conn = sqlite3.connect('chat_history.db')
        conn.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

    # 👇 历史记录列表与删除按钮 (完美缩进)
    for sess in get_all_sessions():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"💬 {sess['title']}", key=f"load_{sess['id']}", use_container_width=True):
                st.session_state.current_session_id = sess['id']
                st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}] + get_messages(
                    sess['id'])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sess['id']}", use_container_width=True):
                delete_session(sess['id'])
                if st.session_state.current_session_id == sess['id']:
                    import uuid

                    st.session_state.current_session_id = str(uuid.uuid4())[:8]
                    st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
                st.toast("已删除该条历史记录")
                st.rerun()
# 6. 主界面与聊天逻辑
# ==========================================
st.markdown('<h1 class="hero-title">🌟 启明 AI 智慧导师</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">当前模式：<b>{persona}</b> | 驱散迷茫，点亮产教融合之路</p >',
            unsafe_allow_html=True)

# 渲染历史消息
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        st.markdown(item["text"])
            else:
                st.markdown(msg["content"])

# 接收输入与 AI 回复
if prompt := st.chat_input("例如：我想做数据分析师怎么学？ / 帮我查一下AI工程师的薪资", key="main_chat_box"):

    current_user_content = [{"type": "text", "text": prompt}]
    if 'uploaded_image' in locals() and uploaded_image is not None:
        base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
        current_user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    st.session_state.messages.append({"role": "user", "content": current_user_content})
    save_message(st.session_state.current_session_id, "user", prompt)

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
        if 'uploaded_image' in locals() and uploaded_image is not None:
            st.image(uploaded_image, width=300)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("导师正在思考..."):

            # RAG 检索
            retrieved_docs = []
            rag_context = ""
            if "vectorstore" in st.session_state and uploaded_image is None:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(prompt)
                if retrieved_docs:
                    rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            current_messages = st.session_state.messages.copy()
            if rag_context:
                current_messages[0]["content"] += f"\n\n【参考教材内容】：\n{rag_context}"

            # 流式输出与 429 重试
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="kimi-k2.6",
                        messages=current_messages,
                        tools=tools,
                        tool_choice="auto",
                        stream=True
                    )

                    full_content = ""
                    tool_calls_dict = {}
                    message_placeholder = st.empty()

                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            full_content += delta.content
                            message_placeholder.markdown(full_content + "▌")
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_dict:
                                    tool_calls_dict[idx] = {"id": tc.id, "type": "function",
                                                            "function": {"name": tc.function.name, "arguments": ""}}
                                if tc.function.arguments:
                                    tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

                    message_placeholder.markdown(full_content)

                    if tool_calls_dict:
                        tool_calls_list = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
                        current_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_list})
                        for tc in tool_calls_list:
                            func_name = tc["function"]["name"]
                            func_args = json.loads(tc["function"]["arguments"])
                            st.toast(f"⚙️ 正在调用：{func_name}...")
                            func_result = available_functions[func_name](**func_args)
                            current_messages.append(
                                {"role": "tool", "tool_call_id": tc["id"], "content": str(func_result)})
                        continue
                    else:
                        if retrieved_docs:
                            with st.expander("🔍 查看参考教材片段"):
                                for i, doc in enumerate(retrieved_docs):
                                    st.info(f"**片段 {i + 1}**: {doc.page_content[:200]}...")

                        st.session_state.messages.append({"role": "assistant", "content": full_content})
                        save_message(st.session_state.current_session_id, "assistant", full_content)
                        break

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "rate_limit" in err_msg:
                        if attempt < max_retries - 1:
                            st.warning(f"⏳ 请求太频繁，正在重试 ({attempt + 1}/{max_retries})...")
                            time.sleep(3)
                            continue
                        else:
                            st.error("💥 账号触发限流，请稍后再试。")
                    else:
                        st.error(f"💥 运行报错：{e}")
                    break

# ===============================
# 7. 开发者调试窗口
# ==========================================
with st.expander("🛠️ 开发者调试窗口"):
    st.write(f"**当前大模型:** Kimi (kimi-k2.6)")
    st.write(f"**历史消息长度:** {len(st.session_state.messages)} 条")
    if "vectorstore" in st.session_state:
        st.success("✅ RAG 本地知识库已就绪")
    else:
        st.warning("⚠️ RAG 本地知识库未加载")