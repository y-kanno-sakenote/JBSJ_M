# modules/analysis/__init__.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from .coauthor import render_coauthor_tab
from .keywords import render_keyword_tab
from .targettype import render_targettype_tab
from .temporal import render_temporal_tab   # ← 追加

def render_analysis_tab(df: pd.DataFrame, use_disk_cache: bool = False) -> None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 共著ネットワーク",
        "⏳ 経年変化",              # ← 追加
        "🧠 キーワード",
        "🏭 対象物・研究タイプ",
    ])
    with tab1:
        render_coauthor_tab(df, use_disk_cache=use_disk_cache)
    with tab2:
        render_temporal_tab(df, use_disk_cache=use_disk_cache)
    with tab3:
        render_keyword_tab(df)
    with tab4:
        render_targettype_tab(df)