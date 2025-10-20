from __future__ import annotations
import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import yearly_keyword_counts
from .base import norm_key, short_preview, get_banner_filters
from .copyui import expander as copy_expander

def render_trend_block(df_use: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns([1,1,1.6,1.6,1.2])
    with c1:
        topn = st.number_input("表示する語数（TopN）", min_value=5, max_value=50, value=15, step=5, key="kw_trend_topn")
    with c2:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="kw_trend_ma")
    with c3:
        include_raw = st.text_input("必須キーワード（部分一致）", value="", key="kw_trend_include", placeholder="例: 酵母, 乳酸菌")
    with c4:
        exclude_raw = st.text_input("除外キーワード（部分一致）", value="", key="kw_trend_exclude", placeholder="例: 試験, 実験")
    with c5:
        metric = st.radio("指標", ["件数","シェア(%)"], index=0, horizontal=True, key="kw_trend_metric")

    yearly = yearly_keyword_counts(df_use)
    include_list = [norm_key(x) for x in _split(include_raw)]
    exclude_list = [norm_key(x) for x in _split(exclude_raw)]

    if not yearly.empty:
        if include_list:
            mask_inc = yearly["keyword"].astype(str).map(lambda v: any(n in norm_key(v) for n in include_list))
            yearly = yearly[mask_inc]
        if exclude_list:
            mask_exc = yearly["keyword"].astype(str).map(lambda v: any(n in norm_key(v) for n in exclude_list))
            yearly = yearly[~mask_exc]

    if yearly.empty:
        st.info("データがありません。"); return

    latest_year = yearly["発行年"].max()
    latest_top = (yearly[yearly["発行年"] == latest_year].sort_values("count", ascending=False)["keyword"].head(int(topn)).tolist())

    piv = (yearly[yearly["keyword"].isin(latest_top)]
           .pivot_table(index="発行年", columns="keyword", values="count", aggfunc="sum")
           .fillna(0).sort_index())

    if metric.startswith("シェア"):
        row_sums = piv.sum(axis=1).replace(0, 1)
        piv = (piv.T / row_sums).T * 100.0

    legend_order = [k for k in latest_top if k in piv.columns]
    others = [k for k in piv.columns if k not in legend_order]
    piv = piv[legend_order + sorted(others)]

    if int(ma) > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    y_label = "シェア(%)" if metric.startswith("シェア") else "件数"
    if HAS_PX:
        fig = px.line(
            piv.reset_index().melt(id_vars="発行年", var_name="キーワード", value_name=y_label),
            x="発行年", y=y_label, color="キーワード", markers=True,
            category_orders={"キーワード": legend_order + sorted(others)}
        )
        if metric.startswith("シェア"):
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(piv)

    # サマリー（期間・対象物・研究タイプを含む）
    y_from, y_to, tg_sel, tp_sel = get_banner_filters(prefix="kw")
    period = f"{int(y_from)}–{int(y_to)}" if y_from is not None and y_to is not None else "—"
    tg_preview = short_preview(tg_sel or [], 3)
    tp_preview = short_preview(tp_sel or [], 3)
    inc_pv = short_preview(include_list, 3)
    exc_pv = short_preview(exclude_list, 3)

    parts = [
        f"条件：表示する語数：{int(topn)}",
        f"移動平均：{int(ma)}年",
        f"指標：{'シェア' if metric.startswith('シェア') else '件数'}",
        f"期間：{period}",
    ]
    if tg_preview:
        parts.append(f"対象物：{tg_preview}")
    if tp_preview:
        parts.append(f"研究タイプ：{tp_preview}")
    if inc_pv:
        parts.append(f"必須：{inc_pv}")
    if exc_pv:
        parts.append(f"除外：{exc_pv}")

    st.caption(" ｜ ".join(parts))
    copy_expander("📋 キーワードをすぐコピー", [c for c in piv.columns if c != "発行年"])

def _split(s: str) -> list[str]:
    import re
    return [w.strip() for w in re.split(r"[,;；、，/／|｜\s　]+", str(s or "")) if w.strip()]