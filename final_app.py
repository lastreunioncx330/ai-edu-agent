# -*- coding: utf-8 -*-
import os
import json
import time
import base64
import sqlite3
import datetime
import random
import uuid
import re
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==========================================
# 1. 页面配置与 UI 极致美化
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
# 2. 数据库操作 & 状态初始化
# ==========================================
def init_db():
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_profile (user_id TEXT PRIMARY KEY, traits TEXT)''')
    conn.commit()
    conn.close()


def save_message(session_id, role, content, app_mode):
    if app_mode == "心理树洞/情绪疏导": return  # 心理模式阅后即焚
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    c = conn.cursor()
    if isinstance(content, list):
        content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
    c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, str(content), datetime.datetime.now()))
    conn.commit()
    conn.close()


def get_messages(session_id):
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


def get_all_sessions():
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "SELECT session_id, MIN(content) FROM messages WHERE role='user' GROUP BY session_id ORDER BY MAX(timestamp) DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1][:15] + "..."} for row in rows]


def delete_session(session_id):
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def save_profile(user_id, traits):
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    conn.execute("INSERT OR REPLACE INTO user_profile (user_id, traits) VALUES (?, ?)", (user_id, traits))
    conn.commit()
    conn.close()


def get_profile(user_id):
    conn = sqlite3.connect('chat_history.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT traits FROM user_profile WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "暂无历史画像（新用户）"


init_db()

# 初始化六维图分数 (全为0)
if "ability_scores" not in st.session_state:
    st.session_state.ability_scores = {"理论基础": 0, "代码实践": 0, "逻辑思维": 0, "数据敏感": 0, "业务理解": 0,
                                       "沟通表达": 0}


def add_ability_score(category, points):
    if category in st.session_state.ability_scores:
        current = st.session_state.ability_scores[category]
        st.session_state.ability_scores[category] = min(100, current + points)


# 清空情绪粉碎机回调函数
def clear_vent_text():
    st.session_state.vent_input = ""


# ==========================================
# 数学公式渲染：$$...$$ 块级 / $...$ 行内，公式走 st.latex，其余走 Markdown
# ==========================================
_BLOCK_MATH = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"\$([^$\n]+)\$")


def _safe_latex(formula: str):
    """公式渲染失败时回退为普通文本，避免整段回答报错。"""
    formula = formula.strip()
    if not formula:
        return
    try:
        st.latex(formula)
    except Exception:
        st.markdown(f"${formula}$")


def render_answer(text: str):
    """把含 $$...$$ / $...$ 的回答拆成 Markdown 段与 LaTeX 段分别渲染。"""
    if not text:
        return
    parts = _BLOCK_MATH.split(text)  # 奇数索引为块级公式
    for i, part in enumerate(parts):
        if part == "":
            continue
        if i % 2 == 1:
            _safe_latex(part)
            continue
        subs = _INLINE_MATH.split(part)  # 普通文本段再拆行内公式
        for j, seg in enumerate(subs):
            if seg == "":
                continue
            if j % 2 == 1:
                _safe_latex(seg)
            else:
                st.markdown(seg)


# ==========================================
# 3. 密码锁
# ==========================================
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.write("##");
    st.write("##")
    st.markdown("<h1 style='text-align: center; color: #2a5298;'>🌟 启明 AI · 专家评审通道</h1>", unsafe_allow_html=True)
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
# 4. 核心配置与工具定义 (含 GraphRAG 与 多智能体)
# ==========================================
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-你的Kimi真实Key")

client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

# 模拟数据库
INDUSTRY_DB = {"数据分析师": {"salary": "15k - 25k", "skills": ["SQL", "Python", "Tableau", "统计学"]},
               "AI大模型工程师": {"salary": "25k - 45k", "skills": ["PyTorch", "LangChain", "RAG"]}}
COLLEGE_DB = {"厦门大学": {"score": "630+", "rank": "全省前5000", "majors": ["会计", "计算机"]},
              "深圳大学": {"score": "600+", "rank": "全省前20000", "majors": ["计算机", "建筑"]}}
STUDENT_PROFILE_DB = {"2023001": "计算机科学专业，GPA 3.8，高数较弱，目标是去字节跳动。"}

# 轻量级 GraphRAG 知识图谱
EDU_KNOWLEDGE_GRAPH = {
    "数据分析师": {"requires_skills": ["SQL", "Python", "统计学"], "related_companies": ["字节跳动", "美团"]},
    "Python": {"is_required_by": ["数据分析师", "AI大模型工程师"],
               "core_concepts": ["列表", "字典", "面向对象", "闭包"]}
}


def graph_rag_retrieve(query: str) -> str:
    context = []
    for node, edges in EDU_KNOWLEDGE_GRAPH.items():
        if node in query:
            context.append(f"【图谱知识】关于『{node}』：")
            for relation, targets in edges.items():
                if isinstance(targets, list): context.append(f"- {relation}: {', '.join(targets)}")
    return "\n".join(context)


# 多智能体协作（规划者 + 审查者）
def multi_agent_collaboration(user_query: str, base_context: str):
    planner_prompt = f"用户问题：{user_query}。背景知识：{base_context}。请简要列出回答该问题的策略（不超过50字）。"
    planner_res = client.chat.completions.create(model="kimi-k2.6",
                                                 messages=[{"role": "user", "content": planner_prompt}])
    plan_draft = planner_res.choices[0].message.content

    critic_prompt = f"你是一个严格的教研主任。请审查以下回答策略是否存在偏差或废话。\n问题：{user_query}\n初稿：{plan_draft}\n请直接输出优化后的最终策略（精简）。"
    critic_res = client.chat.completions.create(model="kimi-k2.6",
                                                messages=[{"role": "user", "content": critic_prompt}])
    return critic_res.choices[0].message.content


# 工具定义
def get_learning_path(major: str, target_job: str) -> str:
    return f"【产业情报】目标岗位《{target_job}》技能需求：SQL、Python等。"


def get_college_info(college_name: str) -> str:
    info = COLLEGE_DB.get(college_name, None)
    return f"【志愿情报局】《{college_name}》录取分: {info['score']}" if info else "暂无数据"


def assess_student_ability(student_id: str, topic: str, score: int) -> str:
    return f"【系统记录】学号{student_id}的《{topic}》成绩({score}分)已入库。"


def generate_quiz(subject: str, difficulty: str) -> str:
    return f"【题库系统】生成了一份《{subject}》{difficulty}难度测试卷。"


tools = [
    {"type": "function", "function": {"name": "get_learning_path", "description": "获取职业薪资与技能",
                                      "parameters": {"type": "object", "properties": {"major": {"type": "string"},
                                                                                      "target_job": {"type": "string"}},
                                                     "required": ["major", "target_job"]}}},
    {"type": "function", "function": {"name": "get_college_info", "description": "查询大学分数线",
                                      "parameters": {"type": "object",
                                                     "properties": {"college_name": {"type": "string"}},
                                                     "required": ["college_name"]}}},
    {"type": "function", "function": {"name": "assess_student_ability", "description": "记录成绩",
                                      "parameters": {"type": "object", "properties": {"student_id": {"type": "string"},
                                                                                      "topic": {"type": "string"},
                                                                                      "score": {"type": "integer"}},
                                                     "required": ["student_id", "topic", "score"]}}},
    {"type": "function", "function": {"name": "generate_quiz", "description": "刷题、要试卷",
                                      "parameters": {"type": "object", "properties": {"subject": {"type": "string"},
                                                                                      "difficulty": {"type": "string"}},
                                                     "required": ["subject", "difficulty"]}}}
]
available_functions = {"get_learning_path": get_learning_path, "get_college_info": get_college_info,
                       "assess_student_ability": assess_student_ability, "generate_quiz": generate_quiz}


# ==========================================
# 4.5 知识库（向量库）清空回调
#     彻底删除底层集合 + 清除相关状态 + 清缓存，
#     保证 rerun 后“预览 / 检索 / 导出”面板随之一并消失。
# ==========================================
def clear_vectorstore():
    vs = st.session_state.pop("vectorstore", None)
    if vs is not None:
        try:
            vs.delete_collection()  # 删掉 Chroma 底层集合，避免残留切片被再次读到
        except Exception:
            pass
    # 清掉知识库相关的一切派生状态
    st.session_state.pop("kb_meta", None)
    st.session_state.pop("kb_search_kw", None)
    # 清掉可能缓存的切片结果，防止预览读到旧数据
    st.cache_data.clear()
    # 标记“刚被清空”，供界面只提示一次
    st.session_state.kb_just_cleared = True


@st.cache_resource(show_spinner="加载向量模型中...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ==========================================
# 5. 侧边栏 (控制台、历史、功能)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #2a5298;'>⚙️ 系统控制台</h2>", unsafe_allow_html=True)

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())[:8]

    st.markdown("#### 🎛️ 工作模式")
    app_mode = st.radio("请选择服务场景：", ["大学学业与职业规划", "高中升学与志愿填报", "心理树洞/情绪疏导"],
                        horizontal=True)

    if app_mode == "大学学业与职业规划":
        persona_options = ["默认全场景导师", "温和的考研导师", "资深的职场HR", "友善的学长/学姐"]
        prompt_map = {"默认全场景导师师": "你是一个全能的AI教育智能体。",
                      "严厉的考研导师": "你是一个极其温和且负责的考研导师。",
                      "资深的职场HR": "你是世界500强企业的资深HR",
                      "友善的学长/学姐": "你是一个有着相关经验且乐于分享的学长/学姐"}
    elif app_mode == "高中升学与志愿填报":
        persona_options = ["资深志愿填报专家", "严厉的高中班主任", "知心的学长/学姐", "高校招生办老师"]
        prompt_map = {"资深志愿填报专家": "你是一个资深高考志愿填报专家。",
                      "温和的高中班主任": "你是一个性格严厉且负责的高三班主任。",
                      "知心的学长/学姐": "你是一个愿意交心的学长/学姐", "高校招生办老师":
                          "你是一个公认名牌高校的招生办老师"}
    else:
        persona_options = ["温柔的心理咨询师", "陪你吐槽的知心树洞", "客观冷静的分析师"]
        prompt_map = {"温柔的心理咨询师": "你是一个温柔的心理辅导老师。若发现严重自残倾向，立刻提供热线 400-161-9995。",
                      "陪你吐槽的知心树洞": "你是一个树洞。",
                      "客观冷静的分析师": "你是一个客观冷静的心理分析师，你不仅提供盲目的情绪安慰，而且用理性的逻辑(如CBT认知行为疗法，ABC情绪模型)当学生倾诉压力时，你必须用理性的逻辑帮他们剖析：A（诱发事件),B(信念/认知),C（情绪结果）。指出他们的认知偏差（如：以偏概全，灾难化情绪),并给出一个具体的心理微行为（如：写情绪日记，正念3分钟），语气专业，克制，有条理"}

    persona = st.selectbox("选择导师角色", persona_options, key="persona_key")
    if persona not in prompt_map:
        persona=list(prompt_map.keys())[0]
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
    if st.session_state.messages[0]["content"] != prompt_map[persona]:
        st.session_state.messages[0]["content"] = prompt_map[persona]
        st.rerun()

    # ==========================================
    # 🌟 模块 A：仅在“非心理模式”下显示 (教务/题库/雷达图)
    # ==========================================
    if app_mode in ["大学学业与职业规划"]:
        st.divider()
        st.markdown("#### 🎓 教务系统绑定")
        student_id_input = st.text_input("输入学号获取专属学情：", placeholder="例如：2023001")
        if student_id_input:
            if student_id_input in STUDENT_PROFILE_DB:
                st.success("✅ 已获取学情数据")
                st.session_state.current_student_info = STUDENT_PROFILE_DB[student_id_input]
            else:
                st.warning("未找到该学号,已切换为通用模式")
                if "current_student_info" in st.session_state: del st.session_state.current_student_info

        st.divider()
        st.markdown("#### 📝 智能题库与学情诊断")

        # 出题来源选择
        quiz_source = st.radio("出题来源", ["通用题库", "基于上传教材生成"], horizontal=True)

        # 🌟 核心修改：只有在通用题库时，才显示科目选择
        if quiz_source == "通用题库":
            quiz_subject = st.selectbox("选择科目", ["Python编程", "高等数学", "数据结构", '英语四级'])
            quiz_difficulty = st.radio("选择难度", ["基础", "进阶", "地狱"], horizontal=True)
        else:
            # 基于教材生成时，不需要选科目，只需选难度
            quiz_subject = "上传的教材"
            quiz_difficulty = st.radio("选择难度", ["基础", "进阶", "地狱"], horizontal=True)

        # 生成按钮逻辑
        if st.button("🎲 生成专属试卷", use_container_width=True):
            if quiz_source == "基于上传教材生成" and "vectorstore" not in st.session_state:
                st.warning("⚠️ 请先在下方上传 PDF 并构建知识库！")
            else:
                if quiz_source == "基于上传教材生成":
                    # 去 RAG 里检索相关知识点
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    context_docs = retriever.invoke("核心知识点 重点 考点")
                    context = "\n".join([d.page_content for d in context_docs])
                    quiz_prompt = f"请严格根据以下我上传的教材内容，为我生成一份【{quiz_difficulty}】难度的测试卷，包含3道选择题。\n【教材内容】：{context}"
                else:
                    quiz_prompt = f"我想做一套【{quiz_subject}】的【{quiz_difficulty}】难度测试卷，请帮我出一份小测。"
                st.session_state.pending_prompt = quiz_prompt
                st.rerun()

        st.markdown("##### 📊 学生六维能力画像")
        categories = list(st.session_state.ability_scores.keys())
        values = list(st.session_state.ability_scores.values())
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself',
                                             line_color='#2a5298'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False,
                          margin=dict(l=40, r=40, t=20, b=20), height=250)
        st.plotly_chart(fig, use_container_width=True)

    # 🌟 模块 B；仅在”志愿填报“下显示
    if app_mode in ["高中升学与志愿填报"]:
        st.divider()
        st.markdown("#### 🎓 教务系统绑定")
        student_id_input = st.text_input("输入学号获取专属学情：", placeholder="例如：2023001")
        if student_id_input:
            if student_id_input in STUDENT_PROFILE_DB:
                st.success("✅ 已获取学情数据")
                st.session_state.current_student_info = STUDENT_PROFILE_DB[student_id_input]
            else:
                st.warning("未找到该学号,已切换为通用模式")
                if "current_student_info" in st.session_state: del st.session_state.current_student_info

        st.divider()
        st.markdown("#### 📝 智能题库与学情诊断")

        quiz_source = st.radio("出题来源", ["通用题库", "基于上传教材生成"], horizontal=True)
        if quiz_source == "通用题库":
            quiz_subject = st.selectbox("选择科目",
                                        ["语文", "数学", "英语", "物理", "化学", "生物", "历史", '政治', "地理"])
            quiz_difficulty = st.radio("选择难度", ["基础", "进阶", "地狱"], horizontal=True)
        else:
            quiz_subject = "上传的教材"
            quiz_difficulty = st.radio("选择难度", ["基础", "进阶", "地狱"], horizontal=True)

        if st.button("🎲 生成专属试卷", use_container_width=True):
            if quiz_source == "基于上传教材生成" and "vectorstore" not in st.session_state:
                st.warning("⚠️ 请先在下方上传 PDF 并构建知识库！")
            else:
                if quiz_source == "基于上传教材生成":
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    context_docs = retriever.invoke("核心知识点 重点 考点")
                    context = "\n".join([d.page_content for d in context_docs])
                    quiz_prompt = f"请严格根据以下我上传的教材内容，为我生成一份【{quiz_difficulty}】难度的测试卷，包含3道选择题。\n【教材内容】：{context}"
                else:
                    quiz_prompt = f"我想做一套【{quiz_subject}】的【{quiz_difficulty}】难度测试卷，请帮我出一份小测。"
                st.session_state.pending_prompt = quiz_prompt
                st.rerun()

        st.markdown("##### 📊 学生六维能力画像")
        categories = list(st.session_state.ability_scores.keys())
        values = list(st.session_state.ability_scores.values())
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself',
                                             line_color='#2a5298'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False,
                          margin=dict(l=40, r=40, t=20, b=20), height=250)
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 🌟 模块 C：仅在“心理模式”下显示 (情绪急救箱)
    # ==========================================
    if app_mode == "心理树洞/情绪疏导":
        st.divider()
        st.error("⚠️ **免责声明**：启明 AI 仅提供倾诉与情绪疏导，**不替代专业的心理咨询与医疗诊断**。若您处于严重危机中，请立即拨打 400-161-9995。")
        st.markdown("#### 🧘 情绪急救箱")
        with st.expander("🫁 478 呼吸放松法"):
            if st.button("▶️ 开始引导", use_container_width=True):
                placeholder = st.empty()
                for _ in range(2):
                    placeholder.info("吸气...");
                    time.sleep(4)
                    placeholder.warning("憋气...");
                    time.sleep(7)
                    placeholder.success("呼气...");
                    time.sleep(8)
                placeholder.success("🎉 完成！,您感觉好点了吗")

        if "show_balloons" not in st.session_state:
            st.session_state.show_balloons = False


        def clear_vent():
            st.session_state.vent_input = ""


        with st.expander("🗑️ 情绪粉碎机"):
            vent_text = st.text_area("写下烦恼，一键粉碎：", height=100, key="vent_input")
            if st.button("💥 粉碎烦恼", use_container_width=True, on_click=clear_vent_text):
                if vent_text or st.session_state.get("vent_input", '') != "":
                    pass
                st.balloons()
                st.success("🎈烦恼已粉碎！让一切随风而去，今天也要开心哦！")
        with st.expander("🎮 舒尔特方格 (转移焦虑)"):
            if "grid_numbers" not in st.session_state:
                nums = list(range(1, 10));
                random.shuffle(nums)
                st.session_state.grid_numbers = nums;
                st.session_state.current_target = 1
            cols = st.columns(3)
            for i in range(9):
                with cols[i % 3]:
                    num = st.session_state.grid_numbers[i]
                    if st.button(str(num), key=f"btn_{num}_{st.session_state.current_target}",
                                 use_container_width=True):
                        if num == st.session_state.current_target:
                            st.session_state.current_target += 1
                            if st.session_state.current_target > 9:
                                st.balloons();
                                st.success("🎉 恭喜完成！")
                                nums = list(range(1, 10));
                                random.shuffle(nums)
                                st.session_state.grid_numbers = nums;
                                st.session_state.current_target = 1
                            st.rerun()
            if st.session_state.current_target <= 9: st.info(f"👉 目标：**{st.session_state.current_target}**")

    # ==========================================
    # 通用模块：PDF 知识库 & 历史记录
    # ==========================================
    if app_mode in ['大学学业与职业规划', '高中升学与志愿填报']:
        st.divider()
        st.markdown("#### 📚 教材知识库")
        uploaded_files = st.file_uploader("上传 PDF", type="pdf", accept_multiple_files=True,
                                          label_visibility="collapsed")

        col_build, col_clear = st.columns([0.6, 0.4])
        with col_build:
            if st.button("构建知识库", use_container_width=True) and uploaded_files:
                with st.spinner("正在解析教材..."):
                    all_docs = []
                    file_names = []
                    for uf in uploaded_files:
                        file_path = f"./temp_{uf.name}"
                        with open(file_path, "wb") as f:
                            f.write(uf.getbuffer())
                        docs = PyPDFLoader(file_path).load()
                        if isinstance(docs, list):
                            all_docs.extend(docs)
                            file_names.append(uf.name)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    if len(all_docs) > 0:
                        splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(
                            all_docs)
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits, embedding=get_embeddings())
                        # 记录知识库元信息，供预览面板展示
                        st.session_state.kb_meta = {
                            "chunk_count": len(splits),
                            "files": file_names,
                        }
                        st.success(f"✅ 知识库构建完成！共切分 {len(splits)} 个文本块")
                        st.rerun()

        with col_clear:
            # 用回调彻底清空：删集合、删状态、清缓存，rerun 后预览面板不再出现
            st.button("清空向量库", use_container_width=True, on_click=clear_vectorstore)
            # 仅在用户刚点过清空时提示一次（pop 即消费，不会常驻）
            if st.session_state.pop("kb_just_cleared", False):
                st.info("🗑️ 向量知识库已清空，预览内容已移除")

        # ========== 知识库预览 / 关键词检索 / 导出（仅在库存在时渲染） ==========
        if "vectorstore" in st.session_state:
            with st.expander("📖 预览 / 检索已入库知识库"):
                kb_meta = st.session_state.get("kb_meta", {})
                chunk_count = kb_meta.get("chunk_count", "?")
                file_list = kb_meta.get("files", [])
                st.info(f"知识库总切片数量：**{chunk_count}**" +
                        (f" ｜ 来源文件：{', '.join(file_list)}" if file_list else ""))

                # 取出全部切片（Chroma 的 get()）
                all_data = st.session_state.vectorstore.get()
                docs_list = all_data.get("documents", []) or []
                metas_list = all_data.get("metadatas", []) or []

                # —— 关键词检索 ——
                kw = st.text_input("关键词检索切片", key="kb_search_kw",
                                   placeholder="输入知识点关键词，如：极限、导数、SQL")
                kw = (kw or "").strip()

                if kw:
                    hits = []
                    for i, (d, m) in enumerate(zip(docs_list, metas_list)):
                        if kw in (d or ""):
                            hits.append((i, d, m))
                    if hits:
                        st.success(f"命中 **{len(hits)}** 个包含「{kw}」的切片")
                        for rank, (i, d, m) in enumerate(hits, start=1):
                            meta = m or {}
                            src_name = meta.get("source", "未知文件").split("/")[-1]
                            page = meta.get("page", "未知页码")
                            st.markdown(f"**命中 {rank} | 切片 #{i + 1} | 文件：{src_name} | 页码：{page}**")
                            st.code(d, language="text")
                            st.divider()
                    else:
                        st.warning(f"未在知识库中检索到包含「{kw}」的切片，换个关键词试试。")
                else:
                    n = len(docs_list)
                    if n > 0:

                        options = list(range(1, min(10, n) + 1))
                        # —— 默认预览前 N 条 ——
                        preview_num = st.selectbox("预览条数", options, index=0)
                    else:
                        preview_num = 0
                        st.warning("暂无检测结果")
                    for i in range(preview_num):
                        meta = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
                        src_name = meta.get("source", "未知文件").split("/")[-1]
                        page = meta.get("page", "未知页码")
                        st.markdown(f"**切片 {i + 1} | 文件：{src_name} | 页码：{page}**")
                        st.code(docs_list[i], language="text")
                        st.divider()

                # —— 导出全部切片为文本 ——
                export_lines = []
                for i, (d, m) in enumerate(zip(docs_list, metas_list), start=1):
                    meta = m or {}
                    src_name = meta.get("source", "未知文件").split("/")[-1]
                    page = meta.get("page", "未知页码")
                    export_lines.append(f"【切片 {i} | {src_name} | 第 {page} 页】\n{d}\n")
                export_text = "\n".join(export_lines)
                st.download_button(
                    label="⬇️ 导出知识库切片文本",
                    data=export_text,
                    file_name="knowledge_chunks.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.divider()
        st.markdown("#### 🖼️ 多模态识图")
        uploaded_images = st.file_uploader("上传图片 (可多选)", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                           label_visibility="collapsed")
        if uploaded_images:
            cols = st.columns(min(len(uploaded_images), 3))
            for idx, img in enumerate(uploaded_images):
                with cols[idx % 3]: st.image(img, use_container_width=True)

    st.divider()
    st.markdown("#### 🗂️ 历史记录")
    if st.button("👑 清空所有历史记录", use_container_width=True):
        conn = sqlite3.connect('chat_history.db', check_same_thread=False)
        conn.execute("DELETE FROM messages");
        conn.commit();
        conn.close()
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

    if st.button("➕ 开启全新对话", use_container_width=True):
        st.session_state.current_session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
        st.rerun()

    for sess in get_all_sessions():
        c1, c2 = st.columns([0.8, 0.2])
        with c1:
            if st.button(f"💬 {sess['title']}", key=f"load_{sess['id']}", use_container_width=True):
                st.session_state.current_session_id = sess['id']
                st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}] + get_messages(
                    sess['id'])
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{sess['id']}", use_container_width=True):
                delete_session(sess['id'])
                if st.session_state.current_session_id == sess['id']:
                    st.session_state.current_session_id = str(uuid.uuid4())[:8]
                    st.session_state.messages = [{"role": "system", "content": prompt_map[persona]}]
                st.rerun()

# ==========================================
# 6. 主界面与聊天逻辑
# ==========================================
st.markdown('<h1 class="hero-title">🌟 启明 AI 智慧导师</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-subtitle">当前模式：<b>{persona}</b> | 驱散迷茫，点亮产教融合之路</p>',
            unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text": render_answer(item["text"])
            else:
                render_answer(msg["content"])

# 接收文字输入
prompt = st.chat_input("例如：我想做数据分析师怎么学？", key="main_chat_box")
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt

if prompt:
    SENSITIVE_WORDS = ["暴力", "色情", "反动", "自杀"]
    if any(word in prompt for word in SENSITIVE_WORDS):
        st.error("⚠️ 您的输入包含违规词汇，已被系统安全拦截。")
        st.stop()

    current_user_content = [{"type": "text", "text": prompt}]
    if 'uploaded_images' in locals() and uploaded_images:
        for img in uploaded_images:
            base64_image = base64.b64encode(img.read()).decode('utf-8')
            img.seek(0)
            current_user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    st.session_state.messages.append({"role": "user", "content": current_user_content})
    save_message(st.session_state.current_session_id, "user", prompt, app_mode)

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
        if 'uploaded_images' in locals() and uploaded_images:
            for img in uploaded_images:
                st.image(img, width=300)

    with st.chat_message("assistant", avatar="🎓"):
        # 1. 多智能体状态展示
        with st.status("🧠 启明多智能体协作系统正在运转...", expanded=True) as status:
            st.write("🔍 [知识引擎] 正在检索图谱与教材库...")
            graph_context = graph_rag_retrieve(prompt)
            retrieved_docs_with_scores = []
            rag_context = ""
            if "vectorstore" in st.session_state and app_mode in ["大学学业与职业规划", "高中升学与志愿填报"]:
                raw_results = st.session_state.vectorstore.similarity_search_with_relevance_scores(prompt, k=3)
                valid_docs = []
                for doc, score in raw_results:
                    if score > 0.4:
                        valid_docs.append((doc, score))
                if valid_docs:
                    for i, (doc, score) in enumerate(valid_docs):
                        page_num = doc.metadata.get('page', '未知')
                        file_name = doc.metadata.get('source', '上传的教材').split("/")[-1]
                        # 文档内容编号
                        rag_context += f"\n[引用{i + 1}](来自{file_name}第{page_num}页：{doc.page_content})\n"
                        retrieved_docs_with_scores.append({"index": i + 1, "doc": doc, "score": score, "page": page_num,
                                                           "file_name": file_name})
            # 混合检索关键词强化
            KEYWORD_PRECISION_DB = {
                "Python": "【精准知识点】Python 核心难点包括：面向对象、闭包、装饰器、生成器。",
                "数据分析": "【精准知识点】数据分析核心三剑客：SQL、Excel、Python(Pandas)。",
                "高数": "【精准知识点】高数重点：微积分、极限、泰勒展开。"
            }
            for kw, context_str in KEYWORD_PRECISION_DB.items():
                if kw in prompt:
                    rag_context += f"\n{context_str}"

            combined_context = f"【图谱知识】\n{graph_context}\n\n【多维检索片段】\n{rag_context}"

            st.write("🧑‍💻 [规划者 Agent] 正在分析意图与制定策略...")
            plan = multi_agent_collaboration(prompt, combined_context)
            st.caption(f"📝 规划思路：{plan}")
            status.update(label="✅ 协作完成，开始生成回答", state="complete", expanded=False)

        # 2. 流式输出与工具调用
        with st.spinner("导师正在执笔..."):
            # 复制当前对话记忆
            current_messages = st.session_state.messages.copy()

            # ==========================================
            # 🌟 第一步：注入“学情”与“长期画像”
            # ==========================================
            if "current_student_info" in st.session_state:
                current_messages[0]["content"] += f"\n\n【学生真实学情】：{st.session_state.current_student_info}"

            user_traits = get_profile(st.session_state.current_session_id)
            current_messages[0]["content"] += f"\n\n【用户长期画像】：{user_traits}"

            # 强制指令：必须结合画像与学情
            current_messages[0]["content"] += "\n\n⚠️【重要】：在回答时，必须结合该学生的真实学情和长期画像，提供个性化建议。绝不能说空话。"

            # ==========================================
            # 🌟 第二步：注入“知识库”与“诊断格式”
            # ==========================================
            if combined_context.strip():
                current_messages[0]["content"] += f"\n\n【多维知识库】：\n{combined_context}\n\n【规划师建议】：\n{plan}"

                # 带有知识库时的回答指令
                current_messages[0]["content"] += """\n\n⚠️【回答指令】：
            1. 请严格基于上面的多维知识库回答用户的问题。如果引用了片段，必须在句末标注 [1]、[2]。
            2. 如果用户询问“怎么学、怎么提升、未来规划”等问题，你必须以【能力诊断报告】的格式输出：
               - 🎯 能力定位：结合学生画像，指出当前最薄弱的环节。
               - 📉 差距分析：对比目标岗位或考试要求，还缺哪些核心技能。
               - 🚀 行动步骤：给出未来 3 天的具体可执行动作（如：刷 5 道题、看某章 PDF）。"""

            elif "vectorstore" in st.session_state:
                # 有库但没搜到
                current_messages[0][
                    "content"] += "\n\n【系统提示】：用户已经上传了相关文件，但当前检索未找到匹配片段。请明确告诉用户你已经读取了文件，但文件中似乎没有提及这个问题。请结合他的学情给出通用建议。"
            else:
                # 根本没上传文件
                current_messages[0][
                    "content"] += "\n\n【系统提示】：用户当前没有上传任何参考文件。请直接用你的常识，结合该学生的学情与画像回答问题，并提示他如果想获得更精准的回答可以上传资料。"
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="kimi-k2.6", messages=current_messages, tools=tools, tool_choice="auto", stream=True
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

                    message_placeholder.empty()
                    render_answer(full_content)

                    if tool_calls_dict:
                        tool_calls_list = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
                        current_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_list})
                        for tc in tool_calls_list:
                            func_name = tc["function"]["name"]
                            func_args = json.loads(tc["function"]["arguments"])
                            st.toast(f"⚙️ 正在调用：{func_name}...")

                            if func_name == "generate_quiz":
                                add_ability_score("理论基础", 5)
                            elif func_name == "assess_student_ability":
                                add_ability_score("代码实践", 10)
                                add_ability_score("数据敏感", 5)

                            func_result = available_functions[func_name](**func_args)
                            current_messages.append(
                                {"role": "tool", "tool_call_id": tc["id"], "content": str(func_result)})
                        continue

                    else:
                        # ==========================================
                        # 🌟 升级版溯源 UI 展示
                        # ==========================================
                        if 'retrieved_docs_with_scores' in locals() and retrieved_docs_with_scores:
                            with st.expander("🔍 查看 AI 参考来源（精准溯源）"):
                                for item in retrieved_docs_with_scores:
                                    st.markdown(
                                        f"**📍 引用 [{item['index']}]** | 来源：`{item['file_name']}` - **第 {item['page']} 页**")
                                    # 把相似度转化为百分比进度条
                                    st.progress(min(float(item['score']), 1.0),
                                                text=f"匹配相关度: {item['score'] * 100:.1f}%")
                                    # 用引用框把原文框起来，显得高级
                                    st.info(f"{item['doc'].page_content.strip()}")
                                    st.write("---")

                        st.session_state.messages.append({"role": "assistant", "content": full_content})
                        save_message(st.session_state.current_session_id, "assistant", full_content, app_mode)
                        add_ability_score("逻辑思维", 1)

                        try:
                            memory_prompt = f"提取特征（30字内）：\n用户:{prompt}\nAI:{full_content}"
                            mem_res = client.chat.completions.create(model="kimi-k2.6", messages=[
                                {"role": "user", "content": memory_prompt}])
                            new_trait = mem_res.choices[0].message.content
                            if "无" not in new_trait:
                                old_trait = get_profile(st.session_state.current_session_id)
                                combined_trait = f"{old_trait}；{new_trait}" if old_trait != "暂无历史画像（新用户）" else new_trait
                                save_profile(st.session_state.current_session_id, combined_trait[:100])
                        except:
                            pass
                        break

                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str:
                        if attempt < max_retries - 1:
                            st.warning(f"⏳ 请求太频繁，等待 20 秒后重试 ({attempt + 1}/{max_retries})...")
                            time.sleep(20)  # 🌟 延长等待时间防限流
                            continue
                        else:
                            st.error("💥 触发限流。")
                    else:
                        st.error(f"💥 报错：{e}")
                    break

# ==========================================
# 7. 开发者调试窗口
# ==========================================
with st.expander("🛠️ 开发者调试窗口"):
    st.write(f"**当前大模型:** Kimi (kimi-k2.6)")
    st.write(f"**历史消息长度:** {len(st.session_state.messages)} 条")
    if app_mode != "心理树洞/情绪疏导":
        st.write(f"**AI 长期记忆画像:** {get_profile(st.session_state.current_session_id)}")
    else:
        st.write(f"**AI 长期记忆画像:** [隐私保护模式：不记录]")
    if "vectorstore" in st.session_state:
        st.success("✅ RAG 本地知识库已就绪")
    else:
        st.warning("⚠️ RAG 本地知识库未加载")
