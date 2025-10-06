# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from .coauthor import render_coauthor_tab
from .keywords import render_keyword_tab
from .targettype import render_targettype_tab

def render_analysis_tab(df: pd.DataFrame) -> None:
#    st.header("📊 分析")
    tab1, tab2, tab3 = st.tabs(["👥 共著ネットワーク", "🧠 キーワード", "🏭 対象物・研究タイプ"])
    with tab1:
        render_coauthor_tab(df)
    with tab2:
        render_keyword_tab(df)
    with tab3:
        render_targettype_tab(df)