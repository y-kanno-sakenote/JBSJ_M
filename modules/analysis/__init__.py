# modules/analysis/__init__.py
import streamlit as st
import pandas as pd

def render_analysis_tab(df: pd.DataFrame) -> None:
    try:
        from .coauthor import render_coauthor_tab
    except Exception as e:
        st.error("❌ 共著ネットワークモジュールの読み込みに失敗しました。")
        st.exception(e)
        return

    tab1, = st.tabs(["👥 共著ネットワーク"])
    with tab1:
        render_coauthor_tab(df)