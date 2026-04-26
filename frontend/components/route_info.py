"""路由策略、推理文本、目标实体、执行步骤条。"""

import streamlit as st

_STRATEGY_STYLE = {
    "local_search":  ("#e6f4ea", "#2d7a3a", "#34a853", "🔎 局部搜索 (KG)"),
    "global_search": ("#e8f0fe", "#1a56a0", "#1a73e8", "🌐 全局搜索 (RAG)"),
    "hybrid":        ("#fff3e0", "#7a4500", "#f57c00", "⚡ 混合搜索"),
}


def render_route_info(result) -> None:
    st.markdown(
        '<span style="font-size:1.1rem;font-weight:700;color:#1a1a2e">🗺️ 路由信息</span>',
        unsafe_allow_html=True,
    )

    bg, tc, border, label = _STRATEGY_STYLE.get(
        result.strategy, ("#f0f0f0", "#555", "#aaa", result.strategy)
    )
    st.markdown(
        f'<span style="display:inline-block;background:{bg};color:{tc};'
        f'border:1px solid {border};padding:4px 14px;border-radius:20px;'
        f'font-size:0.88rem;font-weight:600;margin:8px 0 12px">{label}</span>',
        unsafe_allow_html=True,
    )

    if result.route_reasoning:
        st.markdown(
            f'<div style="font-size:0.88rem;color:#555;background:#f8f9fc;'
            f'border-radius:6px;padding:10px 14px;margin-bottom:12px">'
            f'<b>路由理由：</b>{result.route_reasoning}</div>',
            unsafe_allow_html=True,
        )

    if result.target_entities:
        tags = "".join(
            f'<span style="display:inline-block;background:#e8f0fe;color:#1a56a0;'
            f'border:1px solid #c5d8f8;border-radius:4px;padding:3px 10px;'
            f'font-size:0.82rem;margin:2px 3px">{e}</span>'
            for e in result.target_entities
        )
        st.markdown(
            f'<div style="margin-bottom:12px"><span style="font-size:0.82rem;'
            f'color:#888;margin-right:6px">目标实体</span>{tags}</div>',
            unsafe_allow_html=True,
        )

    # 步骤条：3框2箭，列比例让方框宽、箭头窄
    steps = [
        ("路由",   "🗺️", "#1a73e8"),
        ("检索",   "🔍", "#34a853"),
        ("生成答案","💡", "#f57c00"),
    ]
    cols = st.columns([4, 1, 4, 1, 4])
    for i, (name, icon, color) in enumerate(steps):
        cols[i * 2].markdown(
            f'<div style="text-align:center;background:{color}18;border:2px solid {color}88;'
            f'border-radius:10px;padding:10px 6px">'
            f'<div style="font-size:1.2rem">{icon}</div>'
            f'<div style="font-size:0.88rem;font-weight:700;color:{color};margin-top:4px">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < 2:
            cols[i * 2 + 1].markdown(
                '<div style="text-align:center;font-size:1.6rem;font-weight:900;'
                'color:#666;padding-top:12px;line-height:1">➜</div>',
                unsafe_allow_html=True,
            )
