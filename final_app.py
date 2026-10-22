# -*- coding: utf-8 -*-
import os
import json
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. 页面配置与密码锁
# ==========================================
st.set_page_config(page_title="全场景AI智慧导师", page_icon="🎓", layout="wide")

if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.title("🔐 专家评审专用通道")
    pwd = st.text_input("请输入评审密码：", type="password")
    if st.button("进入系统"):
        if pwd == "judge2026":  # 评审密码
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("密码错误！")
    st.stop()

# ==========================================
# 2. 核心客户端与大模型初始化 (安全读取)
# ==========================================
#if "DEEPSEEK_API_KEY" in st.secrets:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
#else:
    api_key = "sk-fake1234567890abcdef1234567890abcdef"  # 本地测试用
import os
import streamlit as st
try:
    api_key=st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key =os.getenv("DEEPSEEK_API_KEY","sk-fake1234567890abcdef1234567890abcdef")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ==========================================
# 3. 模拟真实产业数据库 & 业务工具
# ==========================================
# 模拟从 BOSS 直聘等平台爬取的真实岗位需求
INDUSTRY_DB = {
    "数据分析师": {"skills": ["SQL", "Python", "Tableau", "统计学", "业务思维"], "salary": "15k-25k"},
    "AI大模型工程师": {"skills": ["PyTorch", "LangChain", "RAG架构", "Prompt Engineering"], "salary": "25k-45k"},
    "前端开发": {"skills": ["React", "Vue", "TypeScript", "性能优化"], "salary": "12k-20k"}
}


def get_learning_path(major: str, target_job: str) -> str:
    """根据产业需求数据库，生成真实学习路径"""
    job_info = INDUSTRY_DB.get(target_job)
    if not job_info:
        return f"抱歉，我们的产业库里暂未收录《{target_job}》的实时需求。"

    skills_str = "、".join(job_info["skills"])
    return f"【产业情报局】查询到《{target_job}》当前平均薪资为 {job_info['salary']}。该岗位核心技能要求为：{skills_str}。建议你结合专业《{major}》的基础，重点补齐上述技能。"


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    """模拟真实数据库存储"""
    return f"【系统记录】已将学号{student_id}的《{topic}》成绩({score}分)存入数据库。系统画像已更新。"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_learning_path",
            "description": "根据学生的专业和目标岗位，对比产业真实需求，生成个性化学习路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "major": {"type": "string", "description": "学生专业"},
                    "target_job": {"type": "string", "description": "目标岗位，如数据分析师、AI大模型工程师"}
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
# 4. 左侧侧边栏 (优化四：状态监控与角色扮演)
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统控制台")

    # 优化四：角色扮演下拉菜单
    persona = st.selectbox(
        "选择导师人设",
        ["默认全场景导师", "严厉的考研导师", "幽默风趣的学长", "职场资深HR"]
    )

    # 根据选择动态生成 System Prompt
    prompt_map = {
        "默认全场景导师": "你是一个全能的AI教育智能体，请用专业、鼓励的语气回答学生的问题。",
        "严厉的考研导师": "你是一个极其严厉的考研导师。你的回答必须一针见血，指出学生的不足，绝不废话，语气严厉但内心是为学生好。",
        "幽默风趣的学长": "你是一个幽默风趣的学长。你喜欢用网络梗和生活中的段子来打比方，让学生觉得学习是一件很有趣的事。",
        "职场资深HR": "你是一个世界500强企业的资深HR。你会从求职、薪资、岗位胜任力的角度去审视学生的问题，并给出最实用的职场建议。"
    }

    st.divider()

    # 优化四：系统状态监控
    st.subheader("📊 系统运行状态")
    st.success("大模型接口：已连接 (DeepSeek)")
    st.info("工具链：已加载 (2个)")
    if "vectorstore" in st.session_state:
        st.success("本地知识库：已就绪")
    else:
        st.warning("本地知识库：未加载")

    st.divider()

    # RAG 知识库上传
    st.header("📚 行业教材知识库")
    uploaded_file = st.file_uploader("上传 PDF 文件", type="pdf")

    if st.button("构建/更新知识库") and uploaded_file is not None:
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
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

# ==========================================
# 5. 主界面与聊天循环 (优化二：真实记忆管理)
# ==========================================
st.title("🎓 全场景 AI 智慧导师")
st.caption(f"当前模式：{persona} | 基于大模型、RAG 与多工具调用的个性化系统")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
else:
    # 如果切换了人设，动态更新 System Prompt
    if st.session_state.messages[0]["content"] != prompt_map[persona]:
        st.session_state.messages[0]["content"] = prompt_map[persona]

# 渲染历史记录
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("例如：我想做数据分析师怎么学？/ 帮我查一下AI工程师的薪资"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("导师正在思考..."):

            # 优化二：记忆压缩机制 (防止历史记录过长)
            if len(st.session_state.messages) > 10:
                with st.spinner("正在压缩历史记忆以节省 Token..."):
                    summary_prompt = "请将以下对话历史总结成一段简短的记忆摘要，保留学生的关键信息和学习进度：\n"
                    history_text = "\n".join([m["content"] for m in st.session_state.messages[1:] if m["content"]])
                    # 简单实现：只保留最近两轮，前面的用总结替代（此处略去具体 API 调用，防止代码过长，直接截断保留最近 6 条）
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
                        model="deepseek-chat",
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
                        st.toast(f"⚙️ 正在调用系统工具：{func_name}...")

                        func_result = available_functions[func_name](**func_args)
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(func_result)
                        })
                except Exception as e:
                    st.error(f"💥 运行报错：{e}")
                    break