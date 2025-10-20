# modules/analysis/targettype_mod/ui_trend.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import yearly_counts
from .base import TARGET_ORDER, TYPE_ORDER, split_multi
from .filters import summary_global_filters

def render_trend_block(df: pd.DataFrame, y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str]) -> None:
    c1, c2, c3, c4 = st.columns([1.5, 1.6, 6.6, 1.5])

    with c1:
        target_mode = st.selectbox(
            "対象",
            ["対象物_top3", "研究タイプ_top3"],
            index=0,
            key="obj_trend_mode",
            format_func=lambda x: "対象物" if x == "対象物_top3" else ("研究タイプ" if x == "研究タイプ_top3" else str(x))
        )

    yearly = yearly_counts(df, target_mode)
    if yearly.empty:
        st.info("データがありません。")
        return

    latest_year = int(yearly["発行年"].max()) if not yearly.empty else None
    auto_top: List[str] = []
    if latest_year is not None:
        auto_top = yearly[yearly["発行年"] == latest_year].sort_values("count", ascending=False)[target_mode].head(5).tolist()

    with c2:
        st.markdown('<div style="height:36px;"></div>', unsafe_allow_html=True)
        auto_top5 = st.checkbox("最新年Top5を自動選択", value=False, key="obj_trend_auto5")
        if "obj_trend_items" not in st.session_state:
            st.session_state["obj_trend_items"] = []

    if auto_top5 and auto_top:
        if st.session_state.get("_obj_trend_autoset") != latest_year:
            st.session_state["obj_trend_items"] = auto_top
            st.session_state["_obj_trend_autoset"] = latest_year

    all_items_raw = sorted({t for v in df.get(target_mode, pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
    if target_mode == "対象物_top3":
        all_items = [x for x in TARGET_ORDER if x in all_items_raw] + [x for x in all_items_raw if x not in TARGET_ORDER]
    else:
        all_items = [x for x in TYPE_ORDER if x in all_items_raw] + [x for x in all_items_raw if x not in TYPE_ORDER]

    if "obj_trend_items" in st.session_state:
        st.session_state["obj_trend_items"] = [x for x in st.session_state["obj_trend_items"] if x in all_items]

    with c3:
        sel = st.multiselect("表示する項目（複数可）", options=all_items[:1000], key="obj_trend_items")

    with c4:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="obj_trend_ma", help="年ごとのノイズをならします。")

    piv = yearly.pivot_table(index="発行年", columns=target_mode, values="count", aggfunc="sum").fillna(0).sort_index()
    if sel:
        piv = piv[[c for c in sel if c in piv.columns]]
    if piv.shape[1] == 0:
        st.info("表示対象がありません。リストから1つ以上選んでください。")
        return
    if ma > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    _sel_key = ",".join(sel) if sel else "__ALL__"
    _uniq_key = f"obj_trend_plot|{target_mode}|{_sel_key}|ma{ma}"

    legend_order = [x for x in (TARGET_ORDER if target_mode == "対象物_top3" else TYPE_ORDER) if x in piv.columns]

    if HAS_PX:
        fig = px.line(piv.reset_index().melt(id_vars="発行年", var_name="項目", value_name="件数"), x="発行年", y="件数", color="項目", markers=True, category_orders={"項目": legend_order})
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
    else:
        st.line_chart(piv, key=_uniq_key)

    _target_label = "対象物" if target_mode == "対象物_top3" else "研究タイプ"
    _shown_n = piv.shape[1]
    st.caption("条件：" + f"対象：{_target_label} ｜ 表示項目数：{_shown_n} ｜ 移動平均：{int(ma)}年 ｜ " + summary_global_filters(y_from, y_to, tg_sel, tp_sel))