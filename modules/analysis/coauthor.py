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
    """共著ネットワークをPyVisで描画"""
    if not (HAS_NX and HAS_PYVIS):
        st.info("⚠️ グラフ描画には networkx / pyvis が必要です。ランキング表は利用できます。")
        return

    edges_use = edges[edges["weight"] >= min_weight]
    if edges_use.empty:
        st.warning("条件に合う共著関係が見つかりませんでした。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        keep = set(top_nodes)
        keep |= {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

# --- HTMLを安全に生成して埋め込み（notebook=False）---
html_path = "coauthor_network.html"
net.write_html(html_path, notebook=False, open_browser=False)

# Streamlitに埋め込み
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()
st.components.v1.html(html, height=height_px, scrolling=True)

# ========= UI構築 =========
def render_coauthor_tab(df: pd.DataFrame):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク）")

    st.caption("共著関係が多いほど、研究ネットワークの中心に位置します。")

    if df is None or "著者" not in df.columns:
        st.warning("著者データが見つかりません。")
        return

    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = int(y.min()), int(y.max()) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        year_from, year_to = st.slider("対象年", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        metric = st.selectbox("スコア計算方式", ["degree", "betweenness", "eigenvector"], index=0)
    with c3:
        top_n = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5)
    with c4:
        min_w = st.number_input("共著回数の下限 (w≥)", min_value=1, max_value=20, value=2, step=1)

    # --- エッジ作成 ---
    edges = build_coauthor_edges(df, year_from, year_to)
    if edges.empty:
        st.info("共著関係が見つかりませんでした。")
        return

    # --- スコア表示 ---
    st.markdown("### 🔝 研究者のつながりランキング")
    rank = _centrality_from_edges(edges, metric=metric).head(int(top_n))
    st.dataframe(rank, use_container_width=True, hide_index=True)

    st.markdown("💬 上位の研究者ほど、他の研究者と多く共著しています。")
    st.caption("＝ネットワークのハブ（情報・技術の中心）を示します。")

    # --- 可視化 ---
    with st.expander("🕸️ ネットワークを可視化してみる", expanded=False):
        st.caption("共著関係をマップ上に可視化します（依存: networkx / pyvis）")
        top_only = st.toggle("トップNの周辺のみ表示（軽量）", value=True)
        top_nodes = rank["著者"].tolist() if top_only else None
        if st.button("🌐 ネットワークを描画する"):
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)