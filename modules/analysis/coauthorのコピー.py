# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化）
- 表: 著者 / 共著数 / つながりスコア の3列
- PyVisは generate_html() に変更（ブラウザ自動起動なし）
- 対象物・研究タイプによるフィルタ機能を復活
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
                         year_to: int | None = None,
                         targets_sel: list[str] | None = None,
                         types_sel: list[str] | None = None) -> pd.DataFrame:
    """著者ペアを抽出して共著回数をカウント（対象物・研究タイプフィルタ対応）"""
    if df is None or "著者" not in df.columns:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    use = df.copy()

    # --- 年範囲フィルタ ---
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # --- 対象物 / 研究タイプフィルタ（部分一致） ---
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    if targets_sel and "対象物_top3" in use.columns:
        keys = [_norm(t) for t in targets_sel]
        use = use[use["対象物_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    if types_sel and "研究タイプ_top3" in use.columns:
        keys = [_norm(t) for t in types_sel]
        use = use[use["研究タイプ_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    # --- 著者ペア生成 ---
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


def _author_coauthor_counts(edges: pd.DataFrame) -> pd.DataFrame:
    """ノードごとの共著数（重み合計）"""
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数"])
    left = edges.groupby("src")["weight"].sum()
    right = edges.groupby("dst")["weight"].sum()
    deg = pd.concat([left, right], axis=1).fillna(0)
    deg["共著数"] = deg.sum(axis=1).astype(int)
    return deg["共著数"].reset_index().rename(columns={"index": "著者"})


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """中心性スコアを算出し、共著数とマージ"""
    counts = _author_coauthor_counts(edges)
    if not HAS_NX:
        out = counts.copy()
        out["つながりスコア"] = out["共著数"].astype(float)
        return out[["著者", "共著数", "つながりスコア"]]

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

    cen_df = pd.Series(cen, name="つながりスコア").reset_index().rename(columns={"index": "著者"})
    out = cen_df.merge(counts, on="著者", how="left")
    out = out.sort_values("つながりスコア", ascending=False).reset_index(drop=True)
    return out[["著者", "共著数", "つながりスコア"]]


def _draw_network(edges: pd.DataFrame, top_nodes=None, min_weight=1, height_px=650):
    """共著ネットワークをPyVisで描画（generate_htmlで埋め込み）"""
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
        existing = [n for n in top_nodes if n in G]
        keep = set(existing) | {nbr for n in existing for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    # 🔧 generate_html() に変更
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========= UI構築 =========
def render_coauthor_tab(df: pd.DataFrame):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク）")

    if df is None or "著者" not in df.columns:
        st.warning("著者データが見つかりません。")
        return

    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin, ymax = int(y.min()), int(y.max())
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    # --- フィルタUI ---
    st.caption("対象年・対象物・研究タイプで絞り込み可能です。")

    c1, c2, c3 = st.columns([1.5, 1.2, 1.2])
    with c1:
        year_from, year_to = st.slider("対象年", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        # 対象物
        targets_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        targets_sel = st.multiselect("対象物（部分一致）", targets_all, default=[])
    with c3:
        # 研究タイプ
        types_all = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        types_sel = st.multiselect("研究タイプ（部分一致）", types_all, default=[])

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        metric = st.selectbox("スコア計算方式", ["degree", "betweenness", "eigenvector"], index=0)
    with c5:
        top_n = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5)
    with c6:
        min_w = st.number_input("共著回数の下限 (w≥)", min_value=1, max_value=20, value=2, step=1)

    # --- エッジ作成 ---
    edges = build_coauthor_edges(df, year_from, year_to, targets_sel, types_sel)
    if edges.empty:
        st.info("共著関係が見つかりませんでした。フィルタを緩めて再度試してください。")
        return

    # --- スコア計算 + 表示 ---
    st.markdown("### 🔝 研究者ランキング（共著数 + スコア）")
    rank = _centrality_from_edges(edges, metric=metric).head(int(top_n))
    st.dataframe(rank, use_container_width=True, hide_index=True)

    # --- ネットワーク可視化 ---
    with st.expander("🕸️ ネットワークを可視化してみる", expanded=False):
        st.caption("共著関係をマップ上に可視化します（依存: networkx / pyvis）")
        top_only = st.toggle("トップNの周辺のみ表示（軽量）", value=True)
        top_nodes = rank["著者"].tolist() if top_only else None
        if st.button("🌐 ネットワークを描画する"):
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)