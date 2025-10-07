# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
経年的変化（共著ネットワークのトレンド）
- 年レンジ＋可変ウィンドウで、共著ネットワークの中心性スコアの推移を可視化
- 表: 著者 / 共著数 / つながりスコア
- グラフ: 時系列折れ線
- アニメ: ウィンドウごとの上位バー（任意）
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# 既存の共著ユーティリティを再利用
from .coauthor import build_coauthor_edges, _AUTHOR_SPLIT_RE, HAS_NX, _centrality_from_edges

def _year_range(df: pd.DataFrame) -> tuple[int, int]:
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025

def _coauthor_count_from_edges(edges: pd.DataFrame) -> pd.Series:
    """各著者の共著回数（重み合計）= 隣接エッジ weight の和"""
    if edges.empty:
        return pd.Series(dtype=float)
    s = edges.groupby("src")["weight"].sum()
    t = edges.groupby("dst")["weight"].sum()
    deg_w = pd.concat([s, t], axis=1).fillna(0).sum(axis=1)
    deg_w.name = "共著数"
    return deg_w

def _window_labels(start: int, end: int, win: int, step: int) -> list[tuple[int, int, str]]:
    """
    例: start=2000, end=2024, win=5, step=3 ->
        [(2000,2004,"2000–2004"), (2003,2007,"2003–2007"), ...]
    """
    labs = []
    y = start
    while y <= end:
        y2 = min(end, y + win - 1)
        labs.append((y, y2, f"{y}–{y2}"))
        y += step
    return labs

def _compute_over_windows(df: pd.DataFrame, y_from: int, y_to: int, win: int, step: int,
                          metric: str) -> pd.DataFrame:
    """ウィンドウごとの中心性と共著数をまとめて返す"""
    results = []
    for s, e, label in _window_labels(y_from, y_to, win, step):
        # 年レンジで切り出し → エッジ化
        edges = build_coauthor_edges(df, s, e)
        if edges.empty:
            continue
        # スコア
        rank = _centrality_from_edges(edges, metric=metric)  # ["著者","つながりスコア", "note"]
        # 共著数
        deg_w = _coauthor_count_from_edges(edges)            # Series index=著者
        rank = rank.merge(deg_w.rename("共著数"), left_on="著者", right_index=True, how="left").fillna({"共著数":0})
        rank["window_label"] = label
        rank["window_start"] = s
        rank["window_end"] = e
        results.append(rank[["window_label","window_start","window_end","著者","共著数","つながりスコア"]])
    if not results:
        return pd.DataFrame(columns=["window_label","window_start","window_end","著者","共著数","つながりスコア"])
    out = pd.concat(results, ignore_index=True)
    # 時系列順で並べる
    out = out.sort_values(["window_start","著者"]).reset_index(drop=True)
    return out

def render_temporal_tab(df: pd.DataFrame) -> None:
    st.markdown("## ⏳ 経年的変化（共著ネットワークのトレンド）")
    st.caption("年レンジとウィンドウ幅を動かすと、誰が中心にいたか・交代したかの推移が見えます。")

    if df is None or "著者" not in df.columns:
        st.warning("著者データが見つかりません。")
        return

    ymin, ymax = _year_range(df)

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        win = st.number_input("ウィンドウ幅（年）", min_value=2, max_value=15, value=5, step=1)
    with c3:
        step = st.number_input("ステップ（年）", min_value=1, max_value=10, value=3, step=1)
    with c4:
        metric = st.selectbox("スコア計算方式", ["degree","betweenness","eigenvector"], index=0,
                              help="networkx未導入時は簡易degree相当で計算")

    with st.spinner("ウィンドウごとに中心性を集計中..."):
        trend = _compute_over_windows(df, year_from, year_to, int(win), int(step), metric=metric)

    if trend.empty:
        st.info("条件に合う共著関係が見つかりませんでした。年レンジやウィンドウ幅を調整してください。")
        return

    # まず、総合TOP（全ウィンドウで最大スコアの高い順）から上位Nを抽出
    c_top, c_anim = st.columns([1,1])
    with c_top:
        top_n = st.number_input("ランキング表示件数", min_value=5, max_value=50, value=20, step=5)
    with c_anim:
        show_anim = st.toggle("ウィンドウごとの上位アニメ表示", value=False)

    # 著者ごとの最大スコアで上位抽出
    top_authors = (trend.groupby("著者")["つながりスコア"]
                   .max().sort_values(ascending=False).head(int(top_n)).index.tolist())
    trend_top = trend[trend["著者"].isin(top_authors)].copy()

    # === 表 ===
    st.markdown("### 🔝 ランキング（ウィンドウ別）")
    # 最新ウィンドウだけの表にする/全ウィンドウにする → 全ウィンドウで出します（理解が進む）
    st.dataframe(
        trend_top.sort_values(["window_start","つながりスコア"], ascending=[True, False]).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

    # === 折れ線 ===
    st.markdown("### 📈 中心性スコアの推移（上位のみ）")
    fig = px.line(
        trend_top,
        x="window_label", y="つながりスコア", color="著者",
        line_group="著者", markers=True,
        hover_data={"共著数":True, "window_start":False, "window_end":False},
    )
    fig.update_layout(xaxis_title="年ウィンドウ", yaxis_title="つながりスコア", legend_title="著者", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # === 任意: アニメーション（ウィンドウごとに上位のバー） ===
    if show_anim:
        st.markdown("### 🎞️ ウィンドウごとの上位バー（アニメ）")
        # 各ウィンドウで上位Nを残す
        frames = []
        for label, g in trend.groupby("window_label"):
            frames.append(g.sort_values("つながりスコア", ascending=False).head(int(top_n)))
        anim_df = pd.concat(frames, ignore_index=True)
        fig_bar = px.bar(
            anim_df,
            x="つながりスコア", y="著者", orientation="h",
            animation_frame="window_label", range_x=[0, anim_df["つながりスコア"].max()*1.1],
            hover_data={"共著数":True},
        )
        fig_bar.update_layout(xaxis_title="つながりスコア", yaxis_title="著者", bargap=0.2)
        st.plotly_chart(fig_bar, use_container_width=True)