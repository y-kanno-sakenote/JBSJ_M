# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd


def render_analysis_tab(df: pd.DataFrame, use_disk_cache: bool = False) -> None:
    # ---- 遅延 import（起動時エラー防止）----
    from .coauthor_entry import render_coauthor_tab
    from .keywords_entry import render_keyword_tab
    from .targettype_entry import render_targettype_tab

    # ---- タブ構成 ----
    tab1, tab2, tab3 = st.tabs([
        "👨‍🔬 研究者",
        "💬 キーワード",
        "🧬 対象物・研究タイプ",
    ])

    with tab1:
        render_coauthor_tab(df)

    with tab2:
        render_keyword_tab(df)

    with tab3:
        render_targettype_tab(df)