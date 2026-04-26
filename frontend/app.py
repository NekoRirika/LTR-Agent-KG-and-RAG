"""Streamlit 主入口：RAG 多智能体可视化问答界面。"""

import sys
from pathlib import Path
from typing import cast

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter import QueryResult, run_query

st.set_page_config(
    page_title="RAG 知识图谱问答",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 15px;
    color: #1a1a2e;
}

/* ── 背景 ── */
.stApp { background: #f0f2f8; }

/* ── 侧边栏 ── */
section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] hr { border-color: #2e2e4e !important; }
section[data-testid="stSidebar"] div.stButton > button {
    background: #2a2a4a !important;
    border: 1px solid #3a3a5a !important;
    color: #c8c8e8 !important;
    font-size: 0.82rem !important;
    padding: 5px 10px !important;
    border-radius: 6px !important;
    text-align: left !important;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #3a3a6a !important;
    color: #ffffff !important;
}

/* ── 标题 ── */
h1 { font-size: 1.7rem !important; font-weight: 700 !important; margin-bottom: 0 !important; }
h2, h3 { font-size: 1.15rem !important; font-weight: 600 !important;
          margin-top: 0.5rem !important; margin-bottom: 0.4rem !important; }

/* ── 按钮 ── */
div.stButton > button {
    font-size: 0.85rem !important;
    padding: 5px 14px !important;
    height: auto !important;
    border-radius: 6px !important;
    transition: all 0.15s !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a73e8, #0d5bba) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(26,115,232,0.35) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1558b0, #0a4a9a) !important;
    box-shadow: 0 4px 12px rgba(26,115,232,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── 输入框 ── */
textarea {
    background: #e4e8f0 !important;
    border: 1.5px solid #c8cfe0 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    color: #1a1a2e !important;
    transition: border-color 0.2s, background 0.2s !important;
}
textarea:focus {
    background: #dce1ec !important;
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
}

/* ── 主内容区白色背景 ── */
.main .block-container {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px 32px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}

/* ── 区域分隔线（灰色粗实线） ── */
.section-divider {
    border: none;
    border-top: 7px solid #c8cdd8;
    margin: 0;
}

/* ── Metric ── */
div[data-testid="stMetric"] {
    background: #f5f7fc;
    border: 1px solid #e2e6f0;
    border-radius: 10px;
    padding: 10px 14px !important;
}
div[data-testid="stMetric"] label { font-size: 0.78rem !important; color: #666 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.25rem !important; font-weight: 700 !important; color: #1a1a2e !important;
}

/* ── Caption ── */
div[data-testid="stCaptionContainer"] { font-size: 0.82rem !important; color: #888 !important; }

/* ── Expander ── */
details { border: 1px solid #e2e6f0 !important; border-radius: 8px !important; margin-bottom: 8px !important; }
details summary {
    font-size: 0.9rem !important; font-weight: 600 !important;
    padding: 9px 14px !important; background: #f8f9fc !important;
    border-radius: 8px !important; color: #333 !important;
}

div[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.9rem !important; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

history = cast(list[QueryResult], st.session_state.history)

# ── 侧边栏（含标题） ──────────────────────────────────────────────────────────

with st.sidebar:
    # 标题固定在侧边栏顶部
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 6px">
      <div style="width:32px;height:32px;border-radius:8px;
                  background:linear-gradient(135deg,#1a73e8,#6c3de8);
                  display:flex;align-items:center;justify-content:center;
                  font-size:0.95rem;flex-shrink:0">🔍</div>
      <div>
        <div style="font-size:0.95rem;font-weight:700;color:#ffffff;line-height:1.2">
          RAG 知识图谱问答
        </div>
        <div style="font-size:0.68rem;color:#8888aa;margin-top:1px">
          LangGraph · Neo4j · 向量检索
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="font-size:0.9rem;font-weight:700;color:#fff;margin:4px 0 4px">🕘 历史查询</p>',
                unsafe_allow_html=True)
    st.divider()
    if not history:
        st.caption("暂无历史记录")
    else:
        for i, r in enumerate(reversed(history)):
            label = r.query[:28] + ("…" if len(r.query) > 28 else "")
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.current_result = r
                st.session_state.input_text = r.query
                st.rerun()

# ── 输入区 ────────────────────────────────────────────────────────────────────

EXAMPLES = [
    ("🔎 局部搜索", "AIGC技术应用于哪些场景？"),
    ("🌐 全局搜索", "有哪些AI治理风险？"),
    ("⚡ 混合搜索", "AIGC对版权归属有哪些影响，整体风险如何？"),
]

st.markdown('<p style="font-size:1.32rem;color:#888;margin-bottom:4px">快速示例</p>',
            unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(ex_cols, EXAMPLES):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.input_text = q
            st.rerun()

query_input = st.text_area(
    "输入问题",
    value=st.session_state.input_text,
    height=100,
    placeholder="请输入您的问题，支持中文…",
    label_visibility="collapsed",
)
submitted = st.button("🚀 提交查询", type="primary")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

if submitted and query_input.strip():
    with st.spinner("正在查询，请稍候…"):
        result = run_query(query_input.strip())
    st.session_state.current_result = result
    st.session_state.input_text = query_input.strip()
    if not history or history[-1].query != result.query:
        history.append(result)
    st.rerun()

# ── 结果展示 ──────────────────────────────────────────────────────────────────

result: QueryResult | None = cast(QueryResult | None, st.session_state.current_result)

if result is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#aaa">
      <div style="font-size:2.5rem;margin-bottom:12px">💬</div>
      <div style="font-size:1rem">在上方输入问题，点击提交开始查询</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if result.error:
    st.error(f"查询出错：{result.error}")
    st.stop()

if not result.kg_results and not result.rag_results:
    st.warning("未检索到任何证据，答案仅供参考。")
elif result.answer_confidence < 0.4:
    st.warning(f"答案置信度较低（{result.answer_confidence:.0%}），建议换个方式提问。")

from components.answer_card import render_answer_card
from components.route_info import render_route_info
from components.evidence import render_evidence
from components.source_docs import render_source_docs
from components.kg_graph import render_kg_graph

def _divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

render_answer_card(result)
_divider()
render_route_info(result)
_divider()
render_evidence(result)

if result.kg_results:
    _divider()
    st.markdown('<span style="font-size:1.05rem;font-weight:700;color:#1a1a2e">🕸️ 知识图谱可视化</span>',
                unsafe_allow_html=True)
    render_kg_graph(result)

_divider()
render_source_docs(result)
