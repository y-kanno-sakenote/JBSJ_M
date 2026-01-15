# modules/common/banners.py
import pandas as pd
import streamlit as st
from .state import GlobalFilters

def render_provenance(df_use: pd.DataFrame, total_n: int, gf: GlobalFilters):
    """出典バナー（検索・分析タブ共通）"""
    try:
        n_filtered = len(df_use) if df_use is not None else 0
        period = f"{gf.year_from}–{gf.year_to}" if gf.year_from and gf.year_to else "—"
        parts = [f"出典：JBSJ DB（N={n_filtered} / {total_n}）", f"期間：{period}"]
        if gf.targets:
            tg_txt = ", ".join(gf.targets[:6]) + (" …" if len(gf.targets) > 6 else "")
            parts.append(f"対象物：{tg_txt}")
        if gf.types:
            tp_txt = ", ".join(gf.types[:6]) + (" …" if len(gf.types) > 6 else "")
            parts.append(f"研究タイプ：{tp_txt}")
        st.caption(" ｜ ".join(parts))
    except Exception:
        st.caption(f"出典：JBSJ DB（N={len(df_use) if df_use is not None else 0} / {total_n}）")

def summarize(gf: GlobalFilters) -> str:
    """条件サマリー文字列（期間・対象物・研究タイプ）"""
    parts = [f"期間：{gf.year_from}–{gf.year_to}"]
    if gf.targets:
        head = ", ".join(gf.targets[:3]) + (" …" if len(gf.targets) > 3 else "")
        parts.append(f"対象物：{head}")
    if gf.types:
        head = ", ".join(gf.types[:3]) + (" …" if len(gf.types) > 3 else "")
        parts.append(f"研究タイプ：{head}")
    return " ｜ ".join(parts)