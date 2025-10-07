# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
経年変化（中心性の移動窓トレンド）

- 指標: degree / betweenness / eigenvector（networkx が無い場合は次数近似でフォールバック）
- 移動窓UIは重複キーを避ける安全版
- Plotly があれば折れ線グラフ、無ければ st.line_chart にフォールバック
"""

from __future__ import annotations
import itertools
import pandas as pd
import streamlit as st

# ---- optional deps ----
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import plotly.express as px  # type: ignore
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ========= 内部ユーティリティ =========
def _year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025


def _split_authors(cell) -> list[str]:
    import re
    if cell is None:
        return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜]+", str(cell)) if w.strip()]


@st.cache_data(ttl=600, show_spinner=False)
def _build_edges(df: pd.DataFrame, y_from: int, y_to: int) -> pd.DataFrame:
    """[y_from, y_to] の範囲で共著エッジを構築"""
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]

    rows = []
    for authors in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(_split_authors(authors)))
        for s, t in itertools.combinations(names, 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.Series:
    """
    エッジから author -> 中心性スコア（Series）を返す
    """
    if edges.empty:
        return pd.Series(dtype=float)

    if not HAS_NX:
        # networkx が無い場合は重み付き次数の簡易版
        deg = (
            pd.concat([edges.groupby("src")["weight"].sum(),
                       edges.groupby("dst")["weight"].sum()], axis=1)
              .fillna(0)
              .sum(axis=1)
              .sort_values(ascending=False)
        )
        deg.name = "score"
        return deg

    # networkx あり：本格計算
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if G.number_of_nodes() == 0:
        return pd.Series(dtype=float)

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    s = pd.Series(cen, dtype=float).sort_values(ascending=False)
    s.name = "score"
    return s


@st.cache_data(ttl=600, show_spinner=False)
def _sliding_window_scores(
    df: pd.DataFrame,
    metric: str,
    start_year: int,
    win: int,
    step: int,
    ymax: int,
) -> pd.DataFrame:
    """
    複数ウィンドウ（start, start+win-1）ごとの中心性スコアを縦持ちDataFrameで返す
    columns: [window, author, score]
    """
    records = []
    s = start_year
    while s <= ymax - win + 1:
        e = s + win - 1
        edges = _build_edges(df, s, e)
        scores = _centrality_from_edges(edges, metric=metric)
        if not scores.empty:
            rec = pd.DataFrame(
                {"window": f"{s}-{e}", "author": scores.index, "score": scores.values}
            )
            records.append(rec)
        s += step

    if not records:
        return pd.DataFrame(columns=["window", "author", "score"])
    return pd.concat(records, ignore_index=True)


# ========= メイン描画 =========
def render_temporal_tab(df: pd.DataFrame, use_disk_cache: bool = True) -> None:
    st.markdown("## ⏳ 経年変化（中心性の移動窓）")

    ymin, ymax = _year_bounds(df)

    # === 年・移動窓コントロール（安全版：固有キー & クランプ） ===
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

        with c1:
            metric = st.selectbox(
                "中心性指標",
                ["degree", "betweenness", "eigenvector"],
                index=0,
                key="temporal_metric",
                help="学術的中心性指標を選択します。networkx未導入の場合は次数近似で計算します。",
            )

        with c2:
            win = st.number_input(
                "移動窓（年）",
                min_value=2,
                max_value=max(2, ymax - ymin + 1),
                value=min(5, max(2, ymax - ymin + 1)),
                step=1,
                key="temporal_win",
                help="例: 5年にすると [開始年, 開始年+4] を1窓として集計します。",
            )

        with c3:
            step = st.number_input(
                "シフト幅（年）",
                min_value=1,
                max_value=max(1, ymax - ymin + 1),
                value=1,
                step=1,
                key="temporal_step",
                help="窓を何年ずつスライドするか。",
            )

        max_start = max(ymin, ymax - int(win) + 1)
        if ymin > max_start:
            # 年範囲 < 窓長 → 窓長を縮める
            win = ymax - ymin + 1
            max_start = ymin
            st.warning("移動窓が年範囲より長いので、自動的に窓長を短縮しました。")

        with c4:
            start_year = st.slider(
                "開始年（移動窓）",
                min_value=ymin,
                max_value=max_start,
                value=min(ymin, max_start),
                step=1,
                key="temporal_start",
                help="この開始年から『移動窓（年）』分を対象に計算します。",
            )

        with c5:
            top_k = st.number_input(
                "表示する著者数",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
                key="temporal_topk",
                help="可視化に含める上位著者数（各窓の上位を総合して選びます）。",
            )

    end_year = start_year + int(win) - 1
    st.caption(f"📅 対象期間: **{start_year} – {end_year}**（{win}年・シフト幅 {step}年）")

    # === 計算 ===
    with st.spinner("時系列スコアを計算中..."):
        scores_long = _sliding_window_scores(
            df=df,
            metric=metric,
            start_year=start_year,
            win=int(win),
            step=int(step),
            ymax=ymax,
        )

    if scores_long.empty:
        st.info("該当期間で共著ネットワークが構成できませんでした。条件を調整してください。")
        return

    # 上位著者を選定（全期間での最大スコア上位）
    top_authors = (
        scores_long.groupby("author")["score"].max().sort_values(ascending=False).head(int(top_k)).index.tolist()
    )
    plot_df = scores_long[scores_long["author"].isin(top_authors)].copy()

    # グラフ描画
    st.markdown("### 中心性スコアの変遷")
    if HAS_PLOTLY:
        fig = px.line(
            plot_df,
            x="window",
            y="score",
            color="author",
            markers=True,
            template="plotly_white",
            title=None,
        )
        fig.update_layout(
            xaxis_title="ウィンドウ（年区間）",
            yaxis_title="中心性スコア",
            legend_title_text="著者",
            height=460,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Pivot → st.line_chart
        pivot = plot_df.pivot(index="window", columns="author", values="score").fillna(0.0)
        st.line_chart(pivot)

    # データ確認用テーブル
    with st.expander("📄 データを表示", expanded=False):
        st.dataframe(
            plot_df.sort_values(["window", "score"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    # コメントのヒント
    st.caption(
        "💡 読み方: ラインが上がる著者はその期間で中心性が高まっています。"
        "ラインが入れ替わるポイントはリーダーシップや共同研究の重心が変化した可能性を示唆します。"
    )