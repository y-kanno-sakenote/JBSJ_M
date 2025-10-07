# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化 + 研究クラスタ）
- 表: 著者 / 共著数 / つながりスコア の3列
- PyVisは generate_html() による埋め込み
- 年・対象物・研究タイプでフィルタ
- コミュニティ検出（クラスタ色分け）＆ クラスタごとの代表研究者／主要キーワード要約
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

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


# ========= エッジ生成（年/対象物/研究タイプでフィルタ）=========
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None,
                         targets_sel: list[str] | None = None,
                         types_sel: list[str] | None = None) -> pd.DataFrame:
    if df is None or "著者" not in df.columns:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    use = df.copy()

    # 年
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # 対象物
    if targets_sel and "対象物_top3" in use.columns:
        keys = [_norm(t) for t in targets_sel]
        use = use[use["対象物_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    # 研究タイプ
    if types_sel and "研究タイプ_top3" in use.columns:
        keys = [_norm(t) for t in types_sel]
        use = use[use["研究タイプ_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    # 著者ペア
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
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数"])
    left = edges.groupby("src")["weight"].sum()
    right = edges.groupby("dst")["weight"].sum()
    deg = pd.concat([left, right], axis=1).fillna(0)
    deg["共著数"] = deg.sum(axis=1).astype(int)
    return deg["共著数"].reset_index().rename(columns={"index": "著者"})


# ========= スコア計算 =========
def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
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


# ========= コミュニティ検出 & 要約 =========
def _detect_communities(edges: pd.DataFrame):
    """コミュニティ（クラスタ）検出。戻り: dict(author -> cluster_id), list of sets"""
    if not HAS_NX or edges.empty:
        return {}, []
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))
    # Greedyモジュラリティ（小〜中規模で安定）
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        # フォールバック：ラベル伝播
        from networkx.algorithms.community import asyn_lpa_communities
        comms = list(asyn_lpa_communities(G, weight="weight"))
    node2cid = {}
    for cid, s in enumerate(comms, 1):
        for n in s:
            node2cid[n] = cid
    return node2cid, comms


def _collect_keywords_for_cluster(df: pd.DataFrame, authors_in_cluster: set[str], top_k: int = 8):
    """クラスタの主要キーワード推定：クラスター著者が含まれる論文から頻出語を抽出"""
    if df is None or not authors_in_cluster:
        return []

    # どの列をキーワード源にするか（存在するものだけ）
    KEY_COLS = [
        "featured_keywords","primary_keywords","secondary_keywords","llm_keywords",
        "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
        "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
    ]
    use_cols = [c for c in KEY_COLS if c in df.columns]

    # クラスタ著者が含まれる行を抽出
    def row_has_author(a_str):
        return any(a in authors_in_cluster for a in split_authors(a_str))

    use = df[df["著者"].fillna("").apply(row_has_author)].copy()
    if use.empty or not use_cols:
        return []

    bag = []
    for c in use_cols:
        for cell in use[c].fillna(""):
            for t in re.split(r"[;；,、，/／|｜\s　]+", str(cell)):
                t = t.strip()
                if t:
                    bag.append(t)

    if not bag:
        return []
    s = pd.Series(bag).value_counts().head(top_k)
    return [f"{k}({v})" for k, v in s.items()]


def _cluster_summary(df: pd.DataFrame, edges: pd.DataFrame, rank_df: pd.DataFrame, top_n_in_cluster=5):
    """クラスタごとの代表研究者（スコア順）と主要キーワード要約を返す"""
    node2cid, comms = _detect_communities(edges)
    if not node2cid:
        return pd.DataFrame(columns=["クラスタ", "代表研究者（上位）", "主要キーワード"])
    # クラスタ→著者リスト
    cluster_rows = []
    for cid, members in enumerate(comms, 1):
        authors = list(members)
        # スコア順で上位を抜粋
        part = rank_df[rank_df["著者"].isin(authors)].sort_values("つながりスコア", ascending=False)
        top_authors = "、".join(part["著者"].head(top_n_in_cluster).tolist()) if not part.empty else ""
        keywords = _collect_keywords_for_cluster(df, set(authors))
        cluster_rows.append({
            "クラスタ": f"C{cid}",
            "代表研究者（上位）": top_authors,
            "主要キーワード": " / ".join(keywords) if keywords else ""
        })
    return pd.DataFrame(cluster_rows)


# ========= ネットワーク描画 =========
def _draw_network(edges: pd.DataFrame, top_nodes=None, min_weight=1, height_px=650, color_by_cluster=True):
    """共著ネットワークをPyVisで描画（コミュニティ色分け, generate_html）"""
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

    # コミュニティ色分け
    node2cid, comms = _detect_communities(edges_use) if color_by_cluster else ({}, [])
    palette = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ab"
    ]

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)

    for n in G.nodes():
        if color_by_cluster and n in node2cid:
            cid = node2cid[n]
            color = palette[(cid-1) % len(palette)]
            net.add_node(n, label=n, color=color)
        else:
            net.add_node(n, label=n)

    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========= UI =========
def render_coauthor_tab(df: pd.DataFrame):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク & クラスタ）")

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
    st.caption("対象年・対象物・研究タイプで絞り込み、共著構造と研究クラスタを可視化します。")

    c1, c2, c3 = st.columns([1.5, 1.2, 1.2])
    with c1:
        year_from, year_to = st.slider("対象年", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        targets_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        targets_sel = st.multiselect("対象物（部分一致）", targets_all, default=[])
    with c3:
        types_all = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        types_sel = st.multiselect("研究タイプ（部分一致）", types_all, default=[])

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        metric = st.selectbox("スコア計算方式", ["degree", "betweenness", "eigenvector"], index=0)
    with c5:
        top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5)
    with c6:
        min_w = st.number_input("共著回数の下限 (w≥)", min_value=1, max_value=20, value=2, step=1)

    # --- エッジ作成 ---
    edges = build_coauthor_edges(df, year_from, year_to, targets_sel, types_sel)
    if edges.empty:
        st.info("共著関係が見つかりませんでした。フィルタを緩めて再度試してください。")
        return

    # --- スコア計算 + 表示 ---
    st.markdown("### 🔝 研究者ランキング")
    rank = _centrality_from_edges(edges, metric=metric).head(int(top_n))
    st.dataframe(rank, use_container_width=True, hide_index=True)

    # --- コミュニティ要約（表） ---
    with st.expander("🧩 研究クラスタ（コミュニティ）要約", expanded=True):
        summary_df = _cluster_summary(df, edges, rank_df=rank, top_n_in_cluster=5)
        if summary_df.empty:
            st.info("コミュニティ検出には networkx が必要です。")
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- ネットワーク可視化（色分け） ---
    with st.expander("🕸️ ネットワークを可視化する（クラスタ色分け）", expanded=False):
        st.caption("共著関係をクラスタ色分けで表示します（依存: networkx / pyvis）")
        top_only = st.toggle("ランキング上位の周辺のみ表示（軽量）", value=True)
        top_nodes = rank["著者"].tolist() if top_only else None
        if st.button("🌐 ネットワークを描画する"):
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700, color_by_cluster=True)