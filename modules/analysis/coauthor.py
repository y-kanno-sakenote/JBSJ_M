# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化）
- ランキング表に「共著数」を追加
- 中心性スコアの棒グラフ + 簡易コメントを表示
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

try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False


# ========= 基本ユーティリティ =========
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell):
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]


@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None,
                         targets_sel: list[str] | None = None,
                         types_sel: list[str] | None = None) -> pd.DataFrame:
    """著者ペアを抽出して共著回数をカウント"""
    if df is None or "著者" not in df.columns:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    use = df.copy()

    # （任意）対象年の絞り込み
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # （任意）対象物 / 研究タイプの部分一致フィルタ
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    if targets_sel and "対象物_top3" in use.columns:
        keys = [_norm(t) for t in targets_sel]
        use = use[use["対象物_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

    if types_sel and "研究タイプ_top3" in use.columns:
        keys = [_norm(t) for t in types_sel]
        use = use[use["研究タイプ_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]

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
    """ノードごとの共著数（重み合計）= 重み付き次数"""
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数"])
    left = edges.groupby("src")["weight"].sum()
    right = edges.groupby("dst")["weight"].sum()
    deg = pd.concat([left, right], axis=1).fillna(0)
    deg["共著数"] = deg.sum(axis=1).astype(int)
    out = deg["共著数"].reset_index().rename(columns={"index": "著者"})
    return out


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """中心性スコアを算出し、共著数とマージして返す"""
    counts = _author_coauthor_counts(edges)  # 著者, 共著数

    # networkx無し：簡易スコア（共著数をそのままスコア扱い）
    if not HAS_NX:
        out = counts.copy()
        out["つながりスコア"] = out["共著数"].astype(float)
        out["note"] = "networkx未導入: 共著数=スコア"
        out = out.sort_values("つながりスコア", ascending=False).reset_index(drop=True)
        return out[["著者", "共著数", "つながりスコア", "note"]]

    # networkxあり
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

    cen_df = (pd.Series(cen, name="つながりスコア")
                .reset_index()
                .rename(columns={"index": "著者"}))

    out = cen_df.merge(counts, on="著者", how="left")
    out["note"] = f"{metric}中心性"
    out = out.sort_values("つながりスコア", ascending=False).reset_index(drop=True)
    return out[["著者", "共著数", "つながりスコア", "note"]]


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
        # グラフに存在するノードだけ使う（存在しないノードで落ちないように）
        existing = [n for n in (top_nodes or []) if n in G]
        keep = set(existing) | {nbr for n in existing for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    # 直接埋め込み（pyvisのブラウザ起動を回避）
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========= UI構築 =========
def render_coauthor_tab(df: pd.DataFrame):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク）")
    st.caption("共著が多い・影響力のある研究者を見つけます。スコアは '中心性'（degree / betweenness / eigenvector）が選べます。")

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

    # 上段：フィルターと計算条件
    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        year_from, year_to = st.slider("対象年", min_value=ymin, max_value=ymax, value=(ymin, ymax))
        metric = st.selectbox("スコア計算方式", ["degree", "betweenness", "eigenvector"], index=0,
                              help="networkx未導入時は「共著数=スコア」の簡易計算になります。")
    with c2:
        # 既存DFの列から候補を収集（部分一致で使う）
        targets_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        types_all   = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in re.split(r"[;；,、，/／|｜\s　]+", str(v)) if t})
        targets_sel = st.multiselect("対象物（部分一致）", targets_all, default=[])
        types_sel   = st.multiselect("研究タイプ（部分一致）", types_all, default=[])

    # --- エッジ作成（年・対象物・研究タイプのフィルタを反映） ---
    edges = build_coauthor_edges(
        df, year_from=year_from, year_to=year_to,
        targets_sel=targets_sel, types_sel=types_sel
    )
    if edges.empty:
        st.info("共著関係が見つかりませんでした。フィルタを緩めて再度試してください。")
        return

    # --- ランキング（共著数を含む） ---
    st.markdown("### 🔝 研究者ランキング（共著数 + スコア）")
    rank = _centrality_from_edges(edges, metric=metric)
    top_n = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5)
    rank_view = rank.head(int(top_n))
    st.dataframe(rank_view[["著者", "共著数", "つながりスコア"]], use_container_width=True, hide_index=True)

    # --- 棒グラフ（中心性スコア） ---
    st.markdown("### 📊 スコアの棒グラフ")
    if not rank_view.empty:
        chart_df = rank_view.copy()
        # 著者名が長い場合に横並びが潰れないように序数付与
        chart_df["label"] = [f"{i+1}. {a}" for i, a in enumerate(chart_df["著者"])]
        if HAS_ALTAIR:
            chart = (alt.Chart(chart_df)
                     .mark_bar()
                     .encode(
                         x=alt.X("つながりスコア:Q", title="中心性スコア"),
                         y=alt.Y("label:N", sort="-x", title="著者"),
                         tooltip=["著者", "共著数", "つながりスコア"]
                     )
                     .properties(height=26 * len(chart_df), width="container"))
            st.altair_chart(chart, use_container_width=True)
        else:
            # 簡易フォールバック
            show = chart_df.set_index("label")["つながりスコア"]
            st.bar_chart(show)

        # --- 簡易インサイト（自動コメント） ---
        top_row = chart_df.iloc[0]
        med = float(chart_df["つながりスコア"].median())
        dominance = "突出" if top_row["つながりスコア"] >= 2.0 * max(med, 1e-12) else "分散"
        hint = {
            "degree": "＝共著相手の多さ（ハブ度）を表します。",
            "betweenness": "＝研究グループ間の“橋渡し役”度合いを表します。",
            "eigenvector": "＝影響力の高い研究者と繋がるほど高くなります。"
        }.get(metric, "")

        st.markdown(
            f"""
- 最上位: **{top_row['著者']}**（共著数: {int(top_row['共著数'])}）  
- スコア分布: **{dominance}傾向**（中央値 ≈ {med:.3f}）  
- 選択スコアの意味: **{metric}** {hint}
            """.strip()
        )

    # --- ネットワーク可視化（任意） ---
    with st.expander("🕸️ ネットワークを可視化（任意・依存あり）", expanded=False):
        st.caption("共著関係をマップ上に可視化します（依存: networkx / pyvis）")
        min_w = st.number_input("表示する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1)
        top_only = st.toggle("テーブル上位の周辺のみ表示（軽量）", value=True)
        top_nodes = rank_view["著者"].tolist() if top_only else None
        if st.button("🌐 ネットワークを描画する"):
            _draw_network(edges, min_weight=int(min_w), top_nodes=top_nodes, height_px=700)