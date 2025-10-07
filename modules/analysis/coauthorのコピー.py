# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（対象年/対象物/研究タイプで絞り込み → ランキング + 可視化）

- フィルタ: 発行年 / 対象物_top3 / 研究タイプ_top3
- 指標: degree / betweenness / eigenvector（選択可能）
- 依存なしでもランキング表は動作
- グラフ描画は networkx + pyvis があれば有効（任意）
"""

import re
import itertools
import pandas as pd
import streamlit as st

# ---- Optional deps ----
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


# ========== ユーティリティ ==========
_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell) -> list[str]:
    if cell is None:
        return []
    return [w.strip() for w in _SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(cell) -> list[str]:
    if not cell:
        return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(cell)) if w.strip()]

@st.cache_data(ttl=600, show_spinner=False)
def _extract_choices(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    tg = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    tp = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    return sorted(tg), sorted(tp)

@st.cache_data(ttl=600, show_spinner=False)
def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    """df（著者列）から共著エッジを作る: ['src','dst','weight']"""
    rows: list[tuple[str, str]] = []
    for a in df.get("著者", pd.Series(dtype=str)).fillna(""):
        names = split_authors(a)
        if len(names) >= 2:
            for s, t in itertools.combinations(sorted(set(names)), 2):
                rows.append((s, t))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    e = pd.DataFrame(rows, columns=["src", "dst"])
    e["pair"] = e.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    e = e.groupby("pair").size().reset_index(name="weight")
    e[["src", "dst"]] = pd.DataFrame(e["pair"].tolist(), index=e.index)
    return e.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)

def _centrality(edges: pd.DataFrame, metric: str) -> pd.DataFrame:
    """中心性（or 簡易スコア）を返す DataFrame[['著者','スコア']]"""
    if edges.empty:
        return pd.DataFrame(columns=["著者", "スコア"])
    if not HAS_NX:
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0)
        deg["スコア"] = deg["weight"].sum(axis=1)
        out = deg["スコア"].sort_values(ascending=False).reset_index()
        out.columns = ["著者", "スコア"]
        return out

    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    out = (pd.Series(cen, name="スコア")
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"index": "著者"}))
    return out

def _draw_network(edges: pd.DataFrame, min_weight: int, top_nodes: list[str] | None, height_px=700):
    """PyVis でグラフ描画（Streamlit向け）"""
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です（ランキングは利用可）。")
        return
    use = edges[edges["weight"] >= int(min_weight)].copy()
    if use.empty:
        st.warning("指定の下限でエッジがありません。閾値を下げてください。")
        return

    G = nx.Graph()
    for _, r in use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        keep = set(top_nodes) | {nbr for n in top_nodes if n in G for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()
        
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110, spring_strength=0.02)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    # Streamlit では show() ではなく write_html() を使う
    net.write_html("coauthor_network.html")
    with open("coauthor_network.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========== メイン UI ==========
def render_coauthor_tab(df: pd.DataFrame):
    st.subheader("👥 共著ネットワーク（研究者どうしのつながり）")

    # 年レンジ
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin, ymax = int(y.min()), int(y.max())
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    # ドロップダウン候補
    tg_choices, tp_choices = _extract_choices(df)

    f1, f2, f3 = st.columns([1.2, 1.2, 1.2])
    with f1:
        year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with f2:
        tg_sel = st.multiselect("対象物（部分一致・複数）", tg_choices, default=[])
    with f3:
        tp_sel = st.multiselect("研究タイプ（部分一致・複数）", tp_choices, default=[])

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        metric = st.selectbox("スコア方式", ["degree", "betweenness", "eigenvector"], index=0,
                              help="degree: つながりの多さ / betweenness: 橋渡し度 / eigenvector: 影響の連鎖")
    with c2:
        top_n = st.number_input("上位表示件数", min_value=5, max_value=100, value=30, step=5)
    with c3:
        min_w = st.number_input("描画用の共著回数下限 (w≥)", min_value=1, max_value=20, value=2, step=1)
    with c4:
        focus_top = st.toggle("ネットワーク描画は上位の周辺だけ", value=True)

    # データをフィルタ
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    if tg_sel and "対象物_top3" in use.columns:
        keys = [s.lower() for s in tg_sel]
        use = use[use["対象物_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    if tp_sel and "研究タイプ_top3" in use.columns:
        keys = [s.lower() for s in tp_sel]
        use = use[use["研究タイプ_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    # エッジ作成 & 指標計算
    edges = build_edges(use)
    if edges.empty:
        st.info("条件に合う共著関係が見つかりませんでした。フィルタを緩めてください。")
        return

    rank = _centrality(edges, metric=metric).head(int(top_n))

    st.markdown("### 🔝 上位研究者（つながりスコア）")
    st.caption(
        "・**degree**: 共同研究の相手が多いほど高スコア\n"
        "・**betweenness**: 研究グループ間の橋渡しを多く担うほど高スコア\n"
        "・**eigenvector**: 影響力のある人とつながるほど高スコア"
    )
    st.dataframe(rank, use_container_width=True, hide_index=True)

    with st.expander("🕸️ ネットワークを可視化する（任意）", expanded=False):
        st.caption("※ networkx / pyvis が導入済みの環境で動作します。")
        if st.button("🌐 描画する"):
            top_nodes = rank["著者"].tolist() if focus_top else None
            _draw_network(edges, min_weight=int(min_w), top_nodes=top_nodes, height_px=700)