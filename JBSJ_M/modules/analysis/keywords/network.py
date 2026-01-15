from __future__ import annotations
import pandas as pd
import streamlit as st

try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network  # type: ignore
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

try:
    from networkx.algorithms.community import greedy_modularity_communities as _greedy_comms  # type: ignore
    HAS_COMMUNITY = True
except Exception:
    HAS_COMMUNITY = False

from .base import PALETTE

@st.cache_data(ttl=600, show_spinner=False)
def compute_node_communities_from_edges(edges: pd.DataFrame) -> dict[str, int]:
    if edges is None or edges.empty or not HAS_NX or not HAS_COMMUNITY: return {}
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1.0)))
    try:
        comms = list(_greedy_comms(G, weight="weight"))
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for gi, nodes in enumerate(comms):
        for n in nodes: mapping[str(n)] = gi
    return mapping

@st.cache_data(ttl=600, show_spinner=False)
def build_pyvis_html(edges: pd.DataFrame, height_px: int = 650, freeze_layout: bool = False) -> tuple[str, str]:
    if not (HAS_NX and HAS_PYVIS): return ("","")
    if edges is None or edges.empty: return ("","")
    import pandas as _pd
    import networkx as _nx
    from pyvis.network import Network as _Network

    G = _nx.Graph()
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"]); w = float(r["weight"])
        if G.has_edge(s, t): G[s][t]["weight"] += w
        else: G.add_edge(s, t, weight=w)
    if G.number_of_nodes()==0: return ("","")

    deg = dict(G.degree())
    deg_w = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        deg_w[u] += w; deg_w[v] += w

    legend_html: str = ""
    if HAS_COMMUNITY:
        from networkx.algorithms.community import greedy_modularity_communities as _greedy
        comms = list(_greedy(G, weight="weight"))
        node_group: dict[str, int] = {}
        for gi, nodes in enumerate(comms):
            for n in nodes: node_group[str(n)] = gi
        chips = []
        for i, nodes in enumerate(comms):
            col = PALETTE[i % len(PALETTE)]
            chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> C{i+1}（{len(nodes)}語）")
        legend_html = " ".join(chips)
    else:
        node_group = {}

    net = _Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-25000, central_gravity=0.15, spring_length=140, spring_strength=0.03, damping=0.18)

    def _scale(v, vmin, vmax, out_min=8, out_max=36):
        if vmax==vmin: return (out_min+out_max)/2
        r = (v - vmin) / (vmax - vmin)
        return out_min + r * (out_max - out_min)

    w_values = list(deg_w.values())
    wmin, wmax = (min(w_values), max(w_values)) if w_values else (0.0, 1.0)

    for n in G.nodes():
        wsum = float(deg_w.get(n, 0.0)); d = int(deg.get(n, 1))
        size = _scale(wsum if wsum>0 else d, wmin if wmin>0 else 0.0, wmax if wmax>0 else 1.0)
        g = node_group.get(str(n))
        if g is not None:
            gi = int(g)
            G.nodes[n]["group"] = gi
            G.nodes[n]["color"] = PALETTE[gi % len(PALETTE)]
        label = str(n)
        label_short = label if len(label)<=18 else (label[:16] + "…")
        G.nodes[n]["label"] = label_short
        G.nodes[n]["title"] = f"{label}&lt;br&gt;重み合計: {wsum:.0f} / 度数: {d}"
        G.nodes[n]["value"] = size

    e_w = [float(d.get("weight", 1.0)) for _,_,d in G.edges(data=True)]
    ew_min, ew_max = (min(e_w), max(e_w)) if e_w else (1.0, 1.0)
    def _ew_scale(w): 
        if ew_max==ew_min: return 1.5
        return 1.0 + 4.0 * (w - ew_min) / (ew_max - ew_min)
    for u,v,d in G.edges(data=True):
        d["width"] = _ew_scale(float(d.get("weight", 1.0)))

    net.from_nx(G)
    try:
        net.set_options("""{"interaction":{"hover":true,"navigationButtons":false,"multiselect":true,"tooltipDelay":120},
                            "nodes":{"shape":"dot","shadow":true,"scaling":{"min":8,"max":36},
                                     "font":{"size":16,"face":"Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial"},
                                     "borderWidth":1},
                            "edges":{"smooth":{"type":"dynamic"},"color":{"opacity":0.45}},
                            "physics":{"stabilization":{"enabled":true,"iterations":220},
                                       "barnesHut":{"avoidOverlap":0.25,"springLength":140,"springConstant":0.03,"damping":0.18},
                                       "minVelocity":0.75}}""")
    except Exception:
        pass

    html = net.generate_html(notebook=False)
    if freeze_layout:
        html = html.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\nnetwork.once('stabilizationIterationsDone', function(){ network.setOptions({ physics:false }); });"
        )
    return (html, legend_html)

def draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650, freeze_layout: bool = False) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("networkx / pyvis が未導入のため、表のみ表示しています。"); return
    if edges.empty:
        st.warning("対象条件でエッジがありません。"); return
    html, legend_html = build_pyvis_html(edges, height_px=height_px, freeze_layout=freeze_layout)
    if not html:
        st.info("ネットワークを生成できませんでした。条件を見直してください。"); return
    st.components.v1.html(html, height=height_px, scrolling=True)
    if legend_html:
        st.markdown("**クラスタ凡例**&nbsp;&nbsp;" + legend_html, unsafe_allow_html=True)
    st.download_button("📥 ネットワークHTMLを保存", data=html.encode("utf-8"),
                       file_name="keyword_cooccurrence_network.html", mime="text/html",
                       key="dl_kw_pyvis_html_cached")