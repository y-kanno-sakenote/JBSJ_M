# modules/analysis/coauthor/network_view.py
from __future__ import annotations
import math
import pandas as pd
import streamlit as st

try:
    import networkx as nx  # type: ignore
    from pyvis.network import Network  # type: ignore
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False

_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
            "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#8c6d31"]

def draw_network(edges: pd.DataFrame,
                 top_nodes: list[str] | None = None,
                 min_weight: int = 1,
                 height_px: int = 650,
                 physics_enabled: bool = True,
                 node_color_map: dict | None = None) -> None:
    if not HAS_GRAPH:
        st.info("グラフ描画には networkx / pyvis が必要です。表は利用できます。")
        return
    edges_use = edges[edges["weight"] >= int(min_weight)].copy()
    if edges_use.empty:
        st.warning("条件に合うエッジがありません。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if top_nodes:
        top_nodes_in = [n for n in top_nodes if n in G]
        keep = set(top_nodes_in)
        for n in top_nodes_in:
            for nbr in G.neighbors(n):
                keep.add(nbr)
        G = G.subgraph(keep).copy()
        if G.number_of_nodes() == 0:
            st.warning("トップNがグラフに存在しません。")
            return

    strength = {}
    for n in G.nodes():
        wsum = 0.0
        for _, _, d in G.edges(n, data=True):
            wsum += float(d.get("weight", 1.0))
        strength[n] = wsum
    label_top = set(sorted(G.nodes(), key=lambda x: strength.get(x, 0.0), reverse=True)[:40])

    local_comm_id = None
    if node_color_map is None:
        try:
            from networkx.algorithms.community import louvain_communities
            comms = list(louvain_communities(G, weight="weight", resolution=1.6))
        except Exception:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G, weight="weight"))
        _tmp = {}
        for i, cset in enumerate(comms):
            for n in cset:
                _tmp[n] = i
        local_comm_id = _tmp

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.set_options(
        f"""
        {{
          "physics": {{
            "enabled": {"true" if physics_enabled else "false"},
            "barnesHut": {{
              "gravitationalConstant": -30000, "centralGravity": 0.25,
              "springLength": 110, "springConstant": 0.02, "damping": 0.30
            }},
            "minVelocity": 0.75, "solver": "barnesHut",
            "stabilization": {{ "enabled": true, "fit": true, "iterations": 800 }}
          }},
          "interaction": {{"hover": true, "tooltipDelay": 120, "zoomView": true, "dragView": true}},
          "nodes": {{"shape": "dot", "borderWidth": 1}},
          "edges": {{"smooth": {{"type": "continuous", "roundness": 0.2}}}}
        }}
        """
    )

    for n in G.nodes():
        wsum = strength.get(n, 0.0)
        size = 8.0 + 4.0 * math.log1p(wsum)
        title = f"{n}｜総共著重み: {int(wsum)}"
        color = None
        if node_color_map is not None:
            color = node_color_map.get(n)
        if color is None:
            cid = 0
            if isinstance(local_comm_id, dict):
                cid = int(local_comm_id.get(n, 0))
            color = _PALETTE[cid % len(_PALETTE)]
        label = n if n in label_top else ""
        net.add_node(n, label=label, title=title, size=size, color=color, borderWidth=0)

    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        width = 1.0 + math.log1p(w)
        title = f"共著回数: {int(w)}"
        net.add_edge(u, v, value=w, width=width, title=title)

    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)
    st.download_button("📥 ネットワークHTML", data=html.encode("utf-8"),
                       file_name="coauthor_network.html", mime="text/html",
                       key="dl_coauthor_html")