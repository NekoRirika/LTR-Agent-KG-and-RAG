"""来源文档列表。"""

import streamlit as st


def render_source_docs(result) -> None:
    st.markdown(
        '<span style="font-size:1.1rem;font-weight:700;color:#1a1a2e">📁 来源文档</span>',
        unsafe_allow_html=True,
    )
    docs = result.source_docs
    if not docs:
        st.info("无来源文档信息")
        return

    items = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'padding:7px 12px;border-radius:6px;margin-bottom:4px;'
        f'background:#f8f9fc;border:1px solid #e2e6f0">'
        f'<span style="font-size:0.82rem;font-weight:700;color:#1a73e8;'
        f'min-width:20px;text-align:center">{i}</span>'
        f'<span style="font-size:0.88rem;color:#333;font-family:monospace">{doc}</span>'
        f'</div>'
        for i, doc in enumerate(docs, 1)
    )
    st.markdown(items, unsafe_allow_html=True)
