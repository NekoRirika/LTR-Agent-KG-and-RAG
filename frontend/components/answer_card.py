"""答案总览卡片。"""

import streamlit as st

_CONF_COLOR = {
    "high":   ("#e6f4ea", "#2d7a3a", "#34a853"),
    "medium": ("#fff8e1", "#7a5c00", "#f9ab00"),
    "low":    ("#fce8e6", "#7a1f1f", "#d93025"),
}


def _conf_level(v: float) -> str:
    if v >= 0.7: return "high"
    if v >= 0.4: return "medium"
    return "low"


def render_answer_card(result) -> None:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">'
        '<span style="font-size:1.1rem;font-weight:700;color:#1a1a2e">💡 回答</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.98rem;line-height:1.8;color:#222;'
        f'background:#f8f9fc;border-left:4px solid #1a73e8;'
        f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:16px">'
        f'{result.answer}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, raw in [
        (c1, "答案置信度", f"{result.answer_confidence:.0%}", result.answer_confidence),
        (c2, "路由置信度", f"{result.route_confidence:.0%}",  result.route_confidence),
        (c3, "KG 证据",   str(len(result.kg_results)),        min(len(result.kg_results)/10, 1.0)),
        (c4, "RAG 证据",  str(len(result.rag_results)),       min(len(result.rag_results)/6, 1.0)),
    ]:
        bg, tc, bar = _CONF_COLOR[_conf_level(raw)]
        col.markdown(
            f'<div style="background:{bg};border:1px solid {bar}33;border-radius:10px;'
            f'padding:12px 14px;text-align:center">'
            f'<div style="font-size:0.75rem;color:{tc};font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:5px">{label}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:{tc}">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
