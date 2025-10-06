# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def render_analysis_tab(df: pd.DataFrame) -> None:
    tab1, tab2, tab3 = st.tabs(["👥 共著ネットワーク", "🧠 キーワード", "🏭 対象物・研究タイプ"])

    with tab1:
        try:
            from .coauthor import render_coauthor_tab
            render_coauthor_tab(df)
        except Exception as e:
            st.error("共著ネットワークの読み込みでエラーが発生しました。")
            st.exception(e)

    with tab2:
        try:
            from .keywords import render_keyword_tab
            render_keyword_tab(df)
        except Exception as e:
            st.info("キーワード分析モジュールが未実装/未配置か、読み込みで失敗しています。")
            st.exception(e)

    with tab3:
        try:
            from .targettype import render_targettype_tab
            render_targettype_tab(df)
        except Exception as e:
            st.info("対象物・研究タイプ分析モジュールが未実装/未配置か、読み込みで失敗しています。")
            st.exception(e)