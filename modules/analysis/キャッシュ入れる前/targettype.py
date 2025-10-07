# modules/analysis/targettype.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def render_targettype_tab(df: pd.DataFrame) -> None:
    st.subheader("🏭 対象物 × 研究タイプ 分析")
    st.info("（プレースホルダ）対象物×研究タイプのヒートマップ、年代別構成比 などを追加予定。")