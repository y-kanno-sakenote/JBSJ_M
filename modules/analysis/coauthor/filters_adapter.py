# modules/analysis/coauthor/filters_adapter.py
from __future__ import annotations
import re
from typing import List, Tuple
import pandas as pd
import streamlit as st
from modules.common.orders import TARGET_ORDER, TYPE_ORDER

# --- 年レンジユーティリティ ---
@st.cache_data(ttl=600, show_spinner=False)
def year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

# --- 分割系 ---
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
_SPLIT_MULTI_RE  = re.compile(r"[;；,、，/／|｜\s　]+")

def split_authors(cell) -> List[str]:
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)

def _sort_with_order(items: List[str], order: List[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(order)}
    return sorted(items, key=lambda x: (order_map.get(x, len(order)), x))

# --- 共通フィルタの呼び出し（外部 or フォールバック） ---
try:
    from modules.common.filters import render_filter_bar as _render_filter_bar  # type: ignore
except Exception:
    def _render_filter_bar(df: pd.DataFrame, key_prefix: str = "authors",
                           show_presets: bool = False, sticky: bool = False):
        ymin, ymax = year_min_max(df)
        tg_all = sorted({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                         for w in split_multi(v) if w.strip()})
        tp_all = sorted({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                         for w in split_multi(v) if w.strip()})
        tg_all = _sort_with_order(list(tg_all), TARGET_ORDER)
        tp_all = _sort_with_order(list(tp_all), TYPE_ORDER)
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            y_from, y_to = st.slider("対象年（範囲）",
                                     min_value=ymin, max_value=ymax,
                                     value=(ymin, ymax),
                                     key=f"{key_prefix}_year")
        with c2:
            tg_sel = st.multiselect("対象物で絞り込み（部分一致）",
                                    options=tg_all, default=[], key=f"{key_prefix}_tg")
        with c3:
            tp_sel = st.multiselect("研究タイプで絞り込み（部分一致）",
                                    options=tp_all, default=[], key=f"{key_prefix}_tp")
        return {"year": (y_from, y_to), "targets": tg_sel, "types": tp_sel}

def adapt_filter_bar(df: pd.DataFrame):
    """filters.render_filter_bar の差異を吸収 -> (df_use, y_from, y_to, tg_sel, tp_sel)"""
    try:
        res = _render_filter_bar(df, key_prefix="authors", show_presets=True, sticky=True)
    except TypeError:
        try:
            res = _render_filter_bar(df, key_prefix="authors")
        except Exception:
            res = df

    if isinstance(res, dict):
        y_from, y_to = res.get("year", year_min_max(df))
        tg_sel = res.get("targets", [])
        tp_sel = res.get("types", [])
        return apply_filters_basic(df, int(y_from), int(y_to), list(tg_sel), list(tp_sel)), int(y_from), int(y_to), list(tg_sel), list(tp_sel)

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

def augment_with_session_state(y_from: int, y_to: int, tg_sel: list[str], tp_sel: list[str], key_prefix="authors"):
    try:
        ss = st.session_state
        if (y_from is None) or (y_to is None):
            yval = ss.get(f"{key_prefix}_year", None)
            if isinstance(yval, (list, tuple)) and len(yval) == 2:
                y_from, y_to = int(yval[0]), int(yval[1])

        def _pick(*names):
            for nm in names:
                v = ss.get(nm, None)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return [str(x) for x in v if str(x).strip()]
            return []
        if not tg_sel:
            tg_sel = _pick(f"{key_prefix}_targets", f"{key_prefix}_target", f"{key_prefix}_tg", f"{key_prefix}_selected_targets")
        if not tp_sel:
            tp_sel = _pick(f"{key_prefix}_types", f"{key_prefix}_type", f"{key_prefix}_tp", f"{key_prefix}_selected_types")
        return int(y_from), int(y_to), tg_sel, tp_sel
    except Exception:
        return y_from, y_to, tg_sel, tp_sel

@st.cache_data(ttl=600, show_spinner=False)
def apply_filters_basic(df: pd.DataFrame, y_from: int, y_to: int, targets: List[str], types: List[str]) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    if targets and "対象物_top3" in use.columns:
        use = use[col_contains_any(use["対象物_top3"], targets)]
    if types and "研究タイプ_top3" in use.columns:
        use = use[col_contains_any(use["研究タイプ_top3"], types)]
    return use