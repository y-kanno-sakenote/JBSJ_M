# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
経年的変化タブ：著者スコアの推移（Plotly が無ければ内蔵チャートに自動フォールバック）
"""
from __future__ import annotations
import itertools
import re
import pandas as pd
import streamlit as st
import networkx as nx

# Plotly は任意依存
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    px = None
    HAS_PLOTLY = False

# --- 共有ユーティリティ（簡易版） ---
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell):
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    rows = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(split_authors(a)))
        for s, t in itertools.combinations(names, 2):
            rows.append((s, t))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = (edges.groupby("pair").size().reset_index(name="weight"))
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    return edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)

import networkx as nx

def _centrality_score(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """中心性指標で著者スコアを算出"""
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    if metric == "degree":
        cen = nx.degree_centrality(G)
    elif metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        raise ValueError("Unknown metric")

    out = pd.Series(cen, name="score").reset_index()
    out.columns = ["author", "score"]
    return out.sort_values("score", ascending=False).reset_index(drop=True)

# --- メイン描画 ---
def render_temporal_tab(df: pd.DataFrame) -> None:
    st.markdown("## ⏱️ 時系列：主要研究者のスコア推移")

    if df is None or "著者" not in df.columns:
        st.warning("著者データが見つかりません。")
        return

    # 年範囲推定
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        years = sorted(y.dropna().astype(int).unique().tolist())
        if years:
            ymin, ymax = min(years), max(years)
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    c1, c2 = st.columns([1, 1])
    with c1:
        win = st.slider("集計ウィンドウ（年幅）", min_value=1, max_value=10, value=5, step=1,
                        help="例: 5なら 2000–2004, 2001–2005 ... とスライドして集計")
    with c2:
        top_k = st.number_input("各ウィンドウの上位N名", min_value=3, max_value=50, value=10, step=1)

    # ウィンドウをスライドしながらスコア計算（簡易 degree スコア）
    records = []
    for start in range(ymin, ymax - win + 2):  # inclusive window
        end = start + win - 1
        edges = build_coauthor_edges(df, start, end)
        if edges.empty:
            continue
        score = _simple_degree_score(edges).head(int(top_k))
        for _, r in score.iterrows():
            records.append({"year": f"{start}–{end}", "author": r["author"], "score": float(r["score"])})

    if not records:
        st.info("条件に合うデータがありませんでした。年幅や年範囲を調整してください。")
        return

    line_df = pd.DataFrame(records)

    st.markdown("### 🔝 ウィンドウごとの上位研究者（簡易スコア）")
    st.dataframe(line_df.sort_values(["year", "score"], ascending=[True, False]),
                 use_container_width=True, hide_index=True)

    st.markdown("### 📈 スコア推移の可視化")
    if HAS_PLOTLY:
        fig = px.line(
            line_df, x="year", y="score", color="author",
            markers=True,
            labels={"year": "年ウィンドウ", "score": "スコア", "author": "著者"},
        )
        fig.update_layout(legend_title_text="著者", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Plotly が未インストールのため内蔵チャートで表示しています。`pip install plotly` でよりリッチな表示になります。")
        # wide 形式にして内蔵 line_chart で表示
        wide = line_df.pivot_table(index="year", columns="author", values="score", fill_value=0)
        st.line_chart(wide)