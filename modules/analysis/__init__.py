# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd


def render_analysis_tab(df: pd.DataFrame, use_disk_cache: bool = False) -> None:
    # ---- 遅延 import（起動時エラー防止）----
    from .coauthor import render_coauthor_tab
    
    try:
        from .keywords_entry import render_keyword_tab
    except Exception as e:
        import traceback
        _kw_err = traceback.format_exc()
        def render_keyword_tab(_df):
            st.error("keywords タブの読み込みに失敗しました。")
            with st.expander("詳細エラー（クリックで展開）", expanded=False):
                st.code(_kw_err, language="python")

    try:
        from .targettype import render_targettype_tab
    except Exception as e:
        import traceback
        _tt_err = traceback.format_exc()
        def render_targettype_tab(_df):
            st.error("targettype タブの読み込みに失敗しました。")
            with st.expander("詳細エラー（クリックで展開）", expanded=False):
                st.code(_tt_err, language="python")

    # ---- タブ構成 ----
    tab1, tab2, tab3 = st.tabs([
        "👨‍🔬 研究者",
        "💬 キーワード",
        "🧬 対象物・研究タイプ",
    ])

    with tab1:
        render_coauthor_tab(df, use_disk_cache=use_disk_cache)

    with tab2:
        render_keyword_tab(df)

    with tab3:
        render_targettype_tab(df)