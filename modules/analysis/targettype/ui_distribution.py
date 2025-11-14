# modules/analysis/targettype_mod/ui_distribution.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

from .compute import count_series
from .filters import summary_global_filters

def _px_bar(df_xy: pd.DataFrame, x_col: str, y_col: str, title: str):
    if not HAS_PX:
        return None
    try:
        fig = px.bar(df_xy, x=x_col, y=y_col, text_auto=True, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, yaxis_title=y_col)
        fig.update_xaxes(tickangle=45, automargin=True)
        return fig
    except Exception:
        return None

def render_distribution_block(df: pd.DataFrame, y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str]) -> None:
    st.markdown("<style>.subttl{font-size:0.95rem; opacity:0.75; margin:0 0 0.25rem;}</style>", unsafe_allow_html=True)

    tg_df = count_series(df, "対象物_top3").reset_index()
    tg_df.columns = ["対象物", "件数"]
    tg_total = int(tg_df["件数"].sum()) if not tg_df.empty else 0

    tp_df = count_series(df, "研究タイプ_top3").reset_index()
    tp_df.columns = ["研究タイプ", "件数"]
    tp_total = int(tp_df["件数"].sum()) if not tp_df.empty else 0

    if tg_df.empty and tp_df.empty:
        st.info("該当データがありません。フィルタを調整してください。")
        return


    c1, c2 = st.columns(2)
    with c1:
        if tg_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            fig = _px_bar(tg_df, "対象物", "件数", f"対象物の出現件数（合計: {tg_total:,}件）")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(tg_df.set_index("対象物")["件数"])

    with c2:
        if tp_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            fig2 = _px_bar(tp_df, "研究タイプ", "件数", f"研究タイプの出現件数（合計: {tp_total:,}件）")
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(tp_df.set_index("研究タイプ")["件数"])

    st.caption("条件：" + summary_global_filters(y_from, y_to, tg_sel, tp_sel))

    # 折り畳み：条件の下に対象物・研究タイプの表を並べて表示（左：対象物、右：研究タイプ）
    with st.expander("📋 対象物／研究タイプの一覧（表）", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            if tg_df.empty:
                st.info("対象物データがありません。")
            else:
                try:
                    st.dataframe(tg_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "📥 対象物表をCSVで保存",
                        data=tg_df.to_csv(index=False).encode("utf-8"),
                        file_name="targettype_counts_by_object.csv",
                        mime="text/csv",
                        key="dl_tg_table_after",
                    )
                except Exception as _e:
                    st.caption(f"表の表示に失敗しました: {_e!s}")

        with cols[1]:
            if tp_df.empty:
                st.info("研究タイプデータがありません。")
            else:
                try:
                    st.dataframe(tp_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "📥 研究タイプ表をCSVで保存",
                        data=tp_df.to_csv(index=False).encode("utf-8"),
                        file_name="targettype_counts_by_type.csv",
                        mime="text/csv",
                        key="dl_tp_table_after",
                    )
                except Exception as _e:
                    st.caption(f"表の表示に失敗しました: {_e!s}")