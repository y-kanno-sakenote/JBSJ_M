# modules/analysis/targettype_mod/ui_cross.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import cross_counts, ordered_index_and_columns
from .base import TARGET_ORDER, TYPE_ORDER
from .filters import summary_global_filters

def render_cross_block(df: pd.DataFrame, y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str]) -> None:
    st.markdown('<div style="font-weight=600; font-size:1.1rem; margin:0 0 0.25rem;">対象物 × 研究タイプ（クロスヒートマップ）</div>', unsafe_allow_html=True)

    cross = cross_counts(df, "対象物_top3", "研究タイプ_top3")
    if cross.empty:
        st.info("クロス集計できるデータがありません。")
        return

    piv = cross.pivot(index="B", columns="A", values="count").fillna(0).astype(int)
    piv.index.name = "研究タイプ"
    piv.columns.name = "対象物"

    idx_order, cols_order = ordered_index_and_columns(piv, TARGET_ORDER, TYPE_ORDER)
    piv = piv.reindex(index=idx_order, columns=cols_order)

    show_values = bool(st.session_state.get("obj_cross_show_values", False))

    if HAS_PX:
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", labels=dict(color="件数"))
        fig.update_xaxes(categoryorder="array", categoryarray=cols_order, tickangle=45, automargin=True)
        fig.update_yaxes(categoryorder="array", categoryarray=idx_order, automargin=True)
        if show_values:
            try:
                fig.update_traces(text=piv.values, texttemplate="%{text}", hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>")
            except Exception:
                fig.update_traces(hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>")
        else:
            fig.update_traces(hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), coloraxis_colorbar_title="件数")
        st.plotly_chart(fig, use_container_width=True)
        rb_spacer, rb_cb = st.columns([6, 1])
        with rb_cb:
            st.checkbox("セルの値を表示", value=show_values, key="obj_cross_show_values", help="ヒートマップの各セルに件数を直接表示します。")
    else:
        st.dataframe(piv, use_container_width=True)
        rb_spacer, rb_cb = st.columns([6, 1])
        with rb_cb:
            st.checkbox("セルの値を表示", value=show_values, key="obj_cross_show_values", help="ヒートマップの各セルに件数を直接表示します。")

    st.caption("条件：" + ("セル値表示：ON ｜ " if bool(st.session_state.get("obj_cross_show_values", False)) else "セル値表示：OFF ｜ ") + summary_global_filters(y_from, y_to, tg_sel, tp_sel))