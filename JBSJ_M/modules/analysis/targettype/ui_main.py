# modules/analysis/targettype_mod/ui_main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
from .filters import adapt_filter_bar_for_obj, augment_with_session_state, render_provenance_banner_from_df
from .ui_distribution import render_distribution_block
from .ui_cross import render_cross_block
from .ui_trend import render_trend_block
from .ui_network import render_cooccurrence_block

def render_targettype_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
          <h2 style="margin:0;">🧬 対象物・研究タイプ分析</h2>
          <span style="opacity:0.8;">対象物・研究タイプの分布・クロス集計・共起ネットワーク・トレンドを確認できます。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    df_use, y_from, y_to, tg_sel, tp_sel = adapt_filter_bar_for_obj(df)
    y_from, y_to, tg_sel, tp_sel = augment_with_session_state(y_from, y_to, tg_sel, tp_sel, key_prefix="obj")
    render_provenance_banner_from_df(df_use, total_n=len(df), y_from=y_from, y_to=y_to, tg_sel=tg_sel, tp_sel=tp_sel)

    tab1, tab2, tab3 = st.tabs(["① 構成比・クロス集計","② 共起ネットワーク","③ トレンド分析"])
    with tab1:
        render_distribution_block(df_use, y_from, y_to, tg_sel, tp_sel)
        st.divider()
        render_cross_block(df_use, y_from, y_to, tg_sel, tp_sel)
    with tab2:
        render_cooccurrence_block(df_use, y_from, y_to, tg_sel, tp_sel)
    with tab3:
        render_trend_block(df_use, y_from, y_to, tg_sel, tp_sel)