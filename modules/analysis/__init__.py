# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import traceback
import streamlit as st
import pandas as pd

def render_analysis_tab(df: pd.DataFrame) -> None:
    st.header("📊 分析")

    # 複数分析タブを定義
    tabs = st.tabs([
        "👥 共著ネットワーク",
        "🧠 キーワード傾向",
        "📈 対象物・年次ヒートマップ"
    ])

    # --- 共著ネットワーク ---
    with tabs[0]:
        try:
            from .coauthor import render_coauthor_tab
            render_coauthor_tab(df)
        except Exception:
            st.error("共著ネットワークの読み込みに失敗しました。")
            st.code(traceback.format_exc())

    # --- キーワード傾向 ---
    with tabs[1]:
        try:
            from .keywords import render_keywords_tab  # ← 今後作成予定
            render_keywords_tab(df)
        except ModuleNotFoundError:
            st.info("🧠 キーワード傾向タブは準備中です。")

    # --- 対象物・年次ヒートマップ ---
    with tabs[2]:
        try:
            from .heatmap import render_heatmap_tab  # ← 今後作成予定
            render_heatmap_tab(df)
        except ModuleNotFoundError:
            st.info("📈 ヒートマップタブは準備中です。")