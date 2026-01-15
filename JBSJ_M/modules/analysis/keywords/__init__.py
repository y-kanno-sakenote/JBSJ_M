# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st

from .filters import safe_filter_bar, render_provenance_banner_from_df, TARGET_ORDER, TYPE_ORDER
from .ui_freq import render_freq_block
from .ui_cooccur import render_cooccur_block
from .ui_trend import render_trend_block

def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
          .kw-header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
          .kw-header h2 { margin:0; }
          .kw-cap { margin:0; font-size:.95rem; color:#6b7280; line-height:1.6; white-space:nowrap; }
          @media (max-width:640px){ .kw-cap{ white-space:normal; } }
        </style>
        <div class="kw-header">
          <h2>💬 キーワード</h2>
          <span class="kw-cap">キーワードの頻度・共起・トレンドを確認できます。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None or df.empty:
        st.info("データが見つかりません。入力を確認してください。")
        return

    df_use = safe_filter_bar(df, key_prefix="kw", target_order=TARGET_ORDER, type_order=TYPE_ORDER)
    y_from = st.session_state.get("kw_year", [None, None])[0] if isinstance(st.session_state.get("kw_year"), (list, tuple)) else None
    y_to   = st.session_state.get("kw_year", [None, None])[1] if isinstance(st.session_state.get("kw_year"), (list, tuple)) else None
    render_provenance_banner_from_df(df_use, total_n=len(df), y_from=y_from, y_to=y_to)

    tab1, tab2, tab3 = st.tabs(["① 頻出キーワード", "② 共起ネットワーク", "③ トレンド分析"])
    with tab1:
        render_freq_block(df_use)
    with tab2:
        render_cooccur_block(df_use)
    with tab3:
        render_trend_block(df_use)