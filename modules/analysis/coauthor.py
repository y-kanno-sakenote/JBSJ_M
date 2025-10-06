# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク：中心性ランキング + ネットワーク可視化（任意）
networkx / pyvis が無い環境でも「ランキング表」は動くようにしてあります。
"""
from __future__ import annotations
import re, itertools
import pandas as pd
import streamlit as st

# オプショナル依存
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

_SPLIT = re.compile(r"[;；,、，/／|｜]+")

def _split_authors(cell) -> list[str]:
    if cell is None: return []
    return [w.strip() for w in _SPLIT.split(str(cell)) if w.strip()]

@st.cache_data(ttl=600, show_spinner=False)
def _build_edges(df: pd.DataFrame, y_from: int|None, y_to: int|None) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns and y_from is not None and y_to is not None:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    rows = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(_split_authors(a)))
        for s, t in itertools.combinations(names, 2):
            rows.append((s, t))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    e = pd.DataFrame(rows, columns=["src","dst"])
    e["pair"] = e.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    e = e.groupby("pair").size().reset_index(name="weight")
    e[["src","dst"]] = pd.DataFrame(e["pair"].tolist(), index=e.index)
    e = e.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return e[["src","dst","weight"]]

def _centrality(edges: pd.DataFrame, metric: str) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["author","centrality","note"])
    if not HAS_NX:
        # 簡易: 重み合計で順位
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0)
        deg["centrality"] = deg.sum(axis=1)
        out = deg["centrality"].sort_values(ascending=False).reset_index()
        out.columns = ["author","centrality"]
        out["note"] = "degree(sum of weights) / no networkx"
        return out
    # networkx があれば本格計算
    import networkx as nx  # type: ignore
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s,t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s,t,weight=w)
    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)
    out = pd.Series(cen, name="centrality").sort_values(ascending=False).reset_index()
    out.columns = ["author","centrality"]
    out["note"] = f"{metric} centrality"
    return out

def _draw_network(edges: pd.DataFrame, top_nodes: list[str] | None, min_weight: int, height_px: int = 700) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です（未導入のため表のみ表示）。")
        return
    import networkx as nx  # type: ignore
    from pyvis.network import Network  # type: ignore

    e2 = edges[edges["weight"] >= min_weight].copy()
    if e2.empty:
        st.warning("条件に合うエッジがありません。")
        return
    G = nx.Graph()
    for _, r in e2.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))
    if top_nodes:
        keep = set(top_nodes) | {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.2, spring_length=110, spring_strength=0.02)
    for n in G.nodes(): net.add_node(n, label=n)
    for s,t,d in G.edges(data=True):
        w = int(d.get("weight",1))
        net.add_edge(s,t,value=w,title=f"共著回数: {w}")
    net.set_options('{"nodes":{"shape":"dot","scaling":{"min":10,"max":40}},"edges":{"smooth":false}}')
    net.show("coauthor_network.html")
    with open("coauthor_network.html","r",encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height_px, scrolling=True)

def render_coauthor_tab(df: pd.DataFrame) -> None:
    st.markdown("### 👥 共著ネットワーク")
    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        metric = st.selectbox("中心性", ["degree","betweenness","eigenvector"], index=0)
    with c3:
        top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5)
    with c4:
        min_w = st.number_input("描画する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1)

    edges = _build_edges(df, y_from, y_to)
    if edges.empty:
        st.info("共著エッジが見つかりませんでした。著者データを確認してください。")
        return

    st.markdown("#### 中心性ランキング")
    rank = _centrality(edges, metric=metric).head(int(top_n))
    st.dataframe(rank, use_container_width=True, hide_index=True)

    with st.expander("🌐 全体ネットワーク（任意）", expanded=False):
        st.caption("※ networkx / pyvis がインストールされている場合のみ描画します。")
        top_only = st.toggle("トップNの周辺だけ可視化（軽量表示）", value=True)
        top_nodes = rank["author"].tolist() if top_only else None
        if st.button("ネットワークを描画する"):
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)