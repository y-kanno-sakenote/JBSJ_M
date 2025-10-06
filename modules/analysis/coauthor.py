# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化）
"""

import re
import itertools
import pandas as pd
import streamlit as st

# --- Optional deps ---
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False


# ========= 基本ユーティリティ =========
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell):
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]


@st.cache_data(ttl=600)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None) -> pd.DataFrame:
    """著者ペアを抽出して共著回数をカウント"""
    if df is None or "著者" not in df.columns:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    use = df.copy()
    if "発行年" in df.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    rows = []
    for authors in use["著者"].fillna(""):
        names = split_authors(authors)
        for s, t in itertools.combinations(sorted(set(names)), 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """中心性スコアを算出"""
    if not HAS_NX:
        deg = pd.concat([
            edges.groupby("src")["weight"].sum(),
            edges.groupby("dst")["weight"].sum(),
        ], axis=1).fillna(0)
        deg["score"] = deg["weight"].sum(axis=1)
        out = deg["score"].sort_values(ascending=False).reset_index()
        out.columns = ["著者", "つながりスコア"]
        out["note"] = "networkx未導入: 簡易スコア"
        return out

    # --- networkxで中心性計算 ---
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight")
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    out = pd.Series(cen, name="つながりスコア").sort_values(ascending=False).reset_index()
    out.columns = ["著者", "つながりスコア"]
    out["note"] = f"{metric}中心性"
    return out


def _draw_network(edges: pd.DataFrame, top_nodes=None, min_weight=1, height_px=650):
    import streamlit as st
    if not (HAS_NX and HAS_PYVIS):
        st.info("⚠️ グラフ描画には networkx / pyvis が必要です。ランキング表は利用できます。")
        return

    edges_use = edges[edges["weight"] >= min_weight]
    if edges_use.empty:
        st.warning("条件に合う共著関係が見つかりませんでした。")
        return

    import networkx as nx
    from pyvis.network import Network

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        keep = set(top_nodes)
        keep |= {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff",
                  font_color="#222", cdn_resources="in_line")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)

    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    net.set_options("""
    {"nodes":{"shape":"dot","scaling":{"min":10,"max":40}},"edges":{"smooth":false}}
    """)

    html_path = "coauthor_network.html"
    net.write_html(html_path, open_browser=False, notebook=False)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height_px, scrolling=True)