# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def render_keyword_tab(df: pd.DataFrame) -> None:
    st.subheader("🧠 キーワード分析")
    st.info("（プレースホルダ）主要語句の頻度推移、共起クラスタ、トピック推定 などを追加予定。")