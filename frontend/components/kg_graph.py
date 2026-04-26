"""KG 三元组 pyvis 关系图可视化。"""

from __future__ import annotations

import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# 实体类型 → 节点颜色
_NODE_COLORS: dict[str, str] = {
    "技术":     "#4A90D9",
    "场景":     "#50C878",
    "风险":     "#E05C5C",
    "治理措施": "#F5A623",
    "主体":     "#9B59B6",
    "政策文件": "#1ABC9C",
    "目标":     "#E67E22",
}
_DEFAULT_COLOR = "#95A5A6"

_PALETTE = list(_NODE_COLORS.values()) + [
    "#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#8E44AD",
]


def _kg_results_to_graph(kg_results: list[dict]) -> dict:
    """将 kg_results 列表转换为 {nodes, links} 格式，按节点名哈希分配颜色组。"""
    node_map: dict[str, dict] = {}
    links: list[dict] = []

    for t in kg_results:
        subj      = str(t.get("subject", "")).strip()
        rel       = str(t.get("relation", "")).strip()
        obj       = str(t.get("object", "")).strip()
        conf      = float(t.get("confidence", 1.0))
        subj_type = str(t.get("subject_type", "")).strip()
        obj_type  = str(t.get("object_type", "")).strip()

        if not subj or not obj:
            continue

        if subj not in node_map:
            node_map[subj] = {"id": subj, "label": subj, "group": subj_type or subj}
        elif subj_type and not node_map[subj]["group"]:
            node_map[subj]["group"] = subj_type

        if obj not in node_map:
            node_map[obj] = {"id": obj, "label": obj, "group": obj_type or obj}
        elif obj_type and not node_map[obj]["group"]:
            node_map[obj]["group"] = obj_type

        links.append({"source": subj, "target": obj, "label": rel, "weight": conf})

    return {"nodes": list(node_map.values()), "links": links}


def render_kg_graph(result) -> None:
    kg = result.kg_results
    if not kg:
        st.info("无 KG 三元组，无法渲染图谱。")
        return

    graph = _kg_results_to_graph(kg)
    nodes = graph["nodes"]
    links = graph["links"]

    if not nodes:
        st.info("三元组数据为空。")
        return

    # ── 显示设置 ──────────────────────────────────────────────────────────────
    with st.expander("图谱显示设置", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            physics = st.checkbox("启用物理引擎", value=True, key="kg_physics")
            node_size = st.slider("节点大小", 10, 50, 22, key="kg_node_size")
        with c2:
            edge_width = st.slider("连接线宽度", 1, 8, 2, key="kg_edge_width")
            spring_len = st.slider("节点间距", 80, 350, 160, key="kg_spring_len")

    # ── 构建 pyvis 网络 ────────────────────────────────────────────────────────
    net = Network(
        height="560px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#333333",
        directed=True,
    )

    net.set_options(f"""
    {{
      "physics": {{
        "enabled": {str(physics).lower()},
        "barnesHut": {{
          "gravitationalConstant": -8000,
          "centralGravity": 0.4,
          "springLength": {spring_len},
          "springConstant": 0.04,
          "damping": 0.15,
          "avoidOverlap": 0.1
        }},
        "stabilization": {{"iterations": 800, "fit": true}}
      }},
      "interaction": {{
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 150,
        "multiselect": true
      }},
      "layout": {{"improvedLayout": true}}
    }}
    """)

    # 为未知 group 动态分配颜色——用节点名哈希保证同名节点颜色一致且各不相同
    seen_groups: dict[str, str] = {}

    for node in nodes:
        name  = node["id"]
        group = node.get("group", "")
        if group in _NODE_COLORS:
            color = _NODE_COLORS[group]
        elif group in seen_groups:
            color = seen_groups[group]
        else:
            # 用 group（或节点名）哈希稳定映射到调色板
            key = group if group else name
            idx = hash(key) % len(_PALETTE)
            color = _PALETTE[idx]
            seen_groups[group] = color

        net.add_node(
            name,
            label=name,
            title=f"{name}（{group}）" if group else name,
            color={
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": color, "border": "#000"},
                "hover":     {"background": color, "border": "#000"},
            },
            size=node_size,
            font={"color": "#ffffff", "size": 13, "face": "Arial"},
            shadow={"enabled": True, "color": "rgba(0,0,0,0.15)", "size": 3},
            borderWidth=2,
        )

    for link in links:
        width = edge_width * min(1 + link["weight"] * 0.3, 2.5)
        net.add_edge(
            link["source"],
            link["target"],
            title=link["label"],
            label=link["label"],
            width=width,
            color={"color": "#aaaaaa", "highlight": "#555", "hover": "#555"},
            smooth={"enabled": True, "type": "dynamic", "roundness": 0.4},
            arrowStrikethrough=False,
        )

    # ── 渲染 HTML ─────────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    components.html(html, height=580)

    # ── 图例 ──────────────────────────────────────────────────────────────────
    all_colors = {**_NODE_COLORS, **seen_groups}
    if all_colors:
        st.markdown("**图例**")
        legend_cols = st.columns(min(len(all_colors), 5))
        for i, (gname, gcolor) in enumerate(all_colors.items()):
            with legend_cols[i % len(legend_cols)]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">'
                    f'<div style="width:14px;height:14px;border-radius:50%;'
                    f'background:{gcolor};flex-shrink:0"></div>'
                    f'<span style="font-size:0.75rem">{gname}</span></div>',
                    unsafe_allow_html=True,
                )
