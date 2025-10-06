# modules/analysis_tab.py
# -*- coding: utf-8 -*-
"""
分析タブ（共起ネットワーク：中心性ランキング + ネットワーク可視化）

- 依存ゼロで「中心性ランキング表」は動作
- networkx / pyvis が入っていれば、インタラクティブな共著ネットワークを描画（任意）
"""

from __future__ import annotations
import re
import itertools
import pandas as pd
import streamlit as st

# ---- 依存はオプショナル ----
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


# ====== 共有ユーティリティ（app.py側と独立に動くよう最小実装） ======
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell) -> list[str]:
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ====== データ加工 ======
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None) -> pd.DataFrame:
    """
    入力: df（少なくとも '著者', '発行年' を含むこと）
    出力: edges DataFrame ['src', 'dst', 'weight']
    """
    use = df.copy()
    # 年で絞り込み（列が無いケースはそのまま）
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # 著者のペアを数える
    rows = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = split_authors(a)
        # 2名以上のときに同一論文内の全ペアを生成
        for s, t in itertools.combinations(sorted(set(names)), 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = (edges.groupby("pair").size()
             .reset_index(name="weight"))
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """
    edges: ['src','dst','weight']
    metric: 'degree'|'betweenness'|'eigenvector'
    """
    # networkx が無い場合は重み付き次数の簡易版を返す
    if not HAS_NX:
        deg = pd.concat([
            edges.groupby("src")["weight"].sum(),
            edges.groupby("dst")["weight"].sum(),
        ], axis=1).fillna(0)
        deg["degree"] = deg["weight"].sum(axis=1)
        out = deg["degree"].sort_values(ascending=False).reset_index()
        out.columns = ["author", "centrality"]
        out["note"] = "degree(sum of co-auth weights) / no networkx"
        return out

    # networkx があるなら本格計算
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
            # 収束しない等の例外時は次数中心性へフォールバック
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    out = (pd.Series(cen, name="centrality")
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"index": "author"}))
    out["note"] = f"{metric} centrality"
    return out


def _draw_network(edges: pd.DataFrame,
                  top_nodes: list[str] | None = None,
                  min_weight: int = 1,
                  height_px: int = 650) -> None:
    """
    pyvis で描画（任意）。依存が無ければスキップ。
    """
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。表は利用できます。")
        return

    # サブグラフ（強いエッジのみ）
    edges_use = edges[edges["weight"] >= min_weight].copy()
    if edges_use.empty:
        st.warning("条件に合うエッジがありません。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        keep = set(top_nodes)
        # トップ＋その隣接を残す
        keep |= {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.2, spring_length=110, spring_strength=0.02)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    net.set_options("""
    {
      "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 40}},
      "edges": {"smooth": false}
    }
    """)
    net.show("coauthor_network.html")
    # 埋め込み表示
    with open("coauthor_network.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height_px, scrolling=True)


# ====== メインの描画関数 ======
def render_analysis_tab(df: pd.DataFrame) -> None:
    st.header("📊 分析：共著ネットワーク")

    # 年の範囲（DFに無い/欠損が多い場合はデフォルト）
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin, ymax = int(y.min()), int(y.max())
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        metric = st.selectbox("中心性", ["degree", "betweenness", "eigenvector"], index=0,
                              help="networkx未導入時は簡易degreeのみ計算")
    with c3:
        top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5)
    with c4:
        min_w = st.number_input("描画する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1)

    # エッジ作成
    edges = build_coauthor_edges(df, year_from, year_to)

    st.markdown("#### 中心性ランキング")
    if edges.empty:
        st.info("共著エッジが見つかりませんでした。著者データを確認してください。")
        return

    rank = _centrality_from_edges(edges, metric=metric)
    rank = rank.head(int(top_n))
    st.dataframe(rank, use_container_width=True, hide_index=True)

    # ネットワーク（任意）
    with st.expander("🌐 全体ネットワーク（任意・依存あり）", expanded=False):
        st.caption("※ networkx / pyvis がインストールされている場合のみ描画します。")
        top_only = st.toggle("トップNの周辺だけ可視化（軽量表示）", value=True)
        top_nodes = rank["author"].tolist() if top_only else None
        draw = st.button("ネットワークを描画する")
        if draw:
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)