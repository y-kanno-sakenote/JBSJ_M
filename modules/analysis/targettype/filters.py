# modules/analysis/targettype_mod/filters.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import streamlit as st
from modules.common.filters import render_filter_bar
from .base import TARGET_ORDER, TYPE_ORDER, apply_filters, year_min_max

def summary_global_filters(y_from: int, y_to: int, tg_sel: list[str] | None, tp_sel: list[str] | None) -> str:
    parts = [f"期間：{int(y_from)}–{int(y_to)}"]
    def _fmt(name: str, vals: list[str] | None, max_items: int = 6) -> None:
        if not vals: return
        vs = [str(v).strip() for v in vals if str(v).strip()]
        if not vs: return
        txt = ", ".join(vs[:max_items]) + (" …" if len(vs) > max_items else "")
        parts.append(f"{name}：{txt}")
    _fmt("対象物", tg_sel)
    _fmt("研究タイプ", tp_sel)
    return " ｜ ".join(parts)

def render_provenance_banner_from_df(df_use: pd.DataFrame, total_n: int, y_from: int | None = None, y_to: int | None = None, tg_sel: list[str] | None = None, tp_sel: list[str] | None = None) -> None:
    try:
        n_filtered = len(df_use) if df_use is not None else 0
        if (y_from is not None) and (y_to is not None):
            period = f"{int(y_from)}–{int(y_to)}"
        else:
            years = pd.to_numeric(df_use.get("発行年", pd.Series(dtype="object")), errors="coerce").dropna().astype(int) if (df_use is not None and "発行年" in df_use.columns) else pd.Series([], dtype=int)
            period = "—" if years.empty else f"{int(years.min())}–{int(years.max())}"
        tg_sel = [t for t in (tg_sel or []) if str(t).strip()]
        tp_sel = [t for t in (tp_sel or []) if str(t).strip()]
        parts = [f"出典：JBSJ DB（N={n_filtered} / {total_n}）", f"期間：{period}"]
        def _fmt_list(name: str, vals: list[str] | None, max_items: int = 6):
            if not vals: return None
            txt = ", ".join(vals[:max_items]) + (" …" if len(vals) > max_items else "")
            return f"{name}：{txt}"
        tg_txt = _fmt_list("対象物", tg_sel)
        tp_txt = _fmt_list("研究タイプ", tp_sel)
        if tg_txt: parts.append(tg_txt)
        if tp_txt: parts.append(tp_txt)
        st.caption(" ｜ ".join(parts))
    except Exception:
        st.caption(f"出典：JBSJ DB（N={len(df_use) if df_use is not None else 0} / {total_n}）")

def adapt_filter_bar_for_obj(df: pd.DataFrame):
    try:
        res = render_filter_bar(df, key_prefix="obj", target_order=TARGET_ORDER, type_order=TYPE_ORDER)
    except TypeError:
        res = render_filter_bar(df, key_prefix="obj")
    if isinstance(res, dict):
        y_from, y_to = res.get("year", year_min_max(df))
        tg_sel = list(res.get("targets", []) or res.get("target", []) or [])
        tp_sel = list(res.get("types", [])   or res.get("type", [])   or [])
        df_use = apply_filters(df, int(y_from), int(y_to), tg_sel, tp_sel)
        return df_use, int(y_from), int(y_to), tg_sel, tp_sel
    if isinstance(res, pd.DataFrame):
        df_use = res
        if "発行年" in df_use.columns:
            y = pd.to_numeric(df_use["発行年"], errors="coerce")
            if y.notna().any():
                y_from, y_to = int(y.min()), int(y.max())
            else:
                y_from, y_to = year_min_max(df)
        else:
            y_from, y_to = year_min_max(df)
        return df_use, y_from, y_to, [], []
    y_from, y_to = year_min_max(df)
    return df, y_from, y_to, [], []

def augment_with_session_state(y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str], key_prefix: str = "obj"):
    try:
        ss = st.session_state
        if (y_from is None) or (y_to is None):
            yval = ss.get(f"{key_prefix}_year", None)
            if isinstance(yval, (list, tuple)) and len(yval) == 2:
                y_from, y_to = int(yval[0]), int(yval[1])
        def _pick_list(*names):
            for nm in names:
                v = ss.get(nm, None)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return [str(x) for x in v if str(x).strip()]
            return []
        if not tg_sel:
            tg_sel = _pick_list(f"{key_prefix}_targets", f"{key_prefix}_target", f"{key_prefix}_tg", f"{key_prefix}_selected_targets")
        if not tp_sel:
            tp_sel = _pick_list(f"{key_prefix}_types", f"{key_prefix}_type", f"{key_prefix}_tp", f"{key_prefix}_selected_types")
        return int(y_from), int(y_to), tg_sel, tp_sel
    except Exception:
        return y_from, y_to, tg_sel, tp_sel