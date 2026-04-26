"""KG 三元组 / RAG 片段分区展示。"""

import pandas as pd
import streamlit as st


def render_evidence(result) -> None:
    st.markdown(
        '<span style="font-size:1.1rem;font-weight:700;color:#1a1a2e">📋 检索证据</span>',
        unsafe_allow_html=True,
    )

    kg  = result.kg_results
    rag = result.rag_results

    # ── KG 三元组 ──────────────────────────────────────────────────────────────
    with st.expander(f"🔗 KG 三元组（{len(kg)} 条）", expanded=bool(kg)):
        if not kg:
            st.info("无 KG 证据")
        else:
            df = pd.DataFrame(kg)
            cols = ["subject", "relation", "object", "confidence", "source_doc", "source_span"]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            df = df[cols].rename(columns={
                "subject": "主体", "relation": "关系", "object": "客体",
                "confidence": "置信度", "source_doc": "来源文档", "source_span": "原文片段",
            })
            df["置信度"] = pd.to_numeric(df["置信度"], errors="coerce").fillna(0)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={"置信度": st.column_config.NumberColumn(format="%.2f")},
            )

    # ── RAG 片段 ───────────────────────────────────────────────────────────────
    with st.expander(f"📄 RAG 文档片段（{len(rag)} 条）", expanded=bool(rag) and not kg):
        if not rag:
            st.info("无 RAG 证据")
        else:
            for i, r in enumerate(rag):
                score = r.get("confidence", r.get("score", ""))
                score_str = f"{float(score):.2f}" if score != "" else "—"
                doc = r.get("source_doc", "未知")
                span = r.get("source_span", "")[:400]
                st.markdown(
                    f'<div style="background:#f8f9fc;border:1px solid #e2e6f0;'
                    f'border-radius:8px;padding:10px 14px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:6px">'
                    f'<span style="font-size:0.82rem;font-weight:600;color:#1a73e8">#{i+1}</span>'
                    f'<span style="font-size:0.78rem;color:#888">'
                    f'📄 {doc} &nbsp;·&nbsp; 相似度 {score_str}</span></div>'
                    f'<div style="font-size:0.88rem;color:#333;line-height:1.6">{span}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
