# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd


def render_analysis_tab(df: pd.DataFrame, use_disk_cache: bool = False) -> None:
    # ---- 遅延 import（起動時エラー防止）----
    from .coauthor import render_coauthor_tab
    try:
        from .temporal import render_temporal_tab
    except Exception:
        def render_temporal_tab(_df, use_disk_cache=False):
            st.warning("temporal タブの読み込みに失敗しました。コードを確認してください。")

    try:
        from .keywords import render_keyword_tab
    except Exception:
        def render_keyword_tab(_df):
            st.warning("keywords タブの読み込みに失敗しました。コードを確認してください。")

    try:
        from .targettype import render_targettype_tab
    except Exception:
        def render_targettype_tab(_df):
            st.warning("targettype タブの読み込みに失敗しました。コードを確認してください。")

    # ---- タブ構成 ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 共著ネットワーク",
        "⏳ 経年変化",
        "🧠 キーワード",
        "🏭 対象物・研究タイプ",
    ])

    with tab1:
        render_coauthor_tab(df, use_disk_cache=use_disk_cache)

    with tab2:
        render_temporal_tab(df)

    with tab3:
        render_keyword_tab(df)

    with tab4:
        render_targettype_tab(df)