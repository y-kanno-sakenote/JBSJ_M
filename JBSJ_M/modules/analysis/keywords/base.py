from __future__ import annotations
import re
from typing import Any, List
import streamlit as st
import pandas as pd


PALETTE = ["#6366F1","#22C55E","#F59E0B","#EF4444","#0EA5E9","#A855F7","#14B8A6","#F97316","#84CC16","#E11D48","#06B6D4","#10B981"]

_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()

def short_preview(items: list[str], maxn: int = 3) -> str:
    vals = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not vals: return ""
    return ", ".join(vals[:maxn]) + (" …" if len(vals) > maxn else "")

def ensure_str_list(values: Any) -> list[str]:
    if values is None: return []
    if isinstance(values, (list, tuple, set)):
        seq = values
    elif hasattr(values, "tolist"):
        try: seq = values.tolist()
        except Exception: seq = [values]
    else:
        seq = [values]
    out: list[str] = []
    for v in seq:
        t = str(v).strip()
        if t: out.append(t)
    return out

def format_year_token(y: Any) -> Any:
    try:
        if isinstance(y, str) and y.isdigit(): return int(y)
        return int(float(y))
    except Exception:
        return y
    
def _split_simple(s: str) -> list[str]:
    return [w.strip() for w in re.split(r"[,;；、，/／|｜\s　]+", str(s or "")) if w.strip()]

def get_banner_filters(prefix: str = "kw") -> tuple[int|None, int|None, list[str], list[str]]:
    """
    共通フィルターUIで選択された『年レンジ／対象物／研究タイプ』を取得して返す。
    return: (y_from, y_to, targets, types)
    """
    # 年レンジ（スライダー値）
    y_from = y_to = None
    yv = st.session_state.get(f"{prefix}_year")
    if isinstance(yv, (list, tuple)) and len(yv) == 2:
        try:
            y_from, y_to = int(yv[0]), int(yv[1])
        except Exception:
            y_from = y_to = None

    # 対象物・研究タイプ（セッションに入っている代表キーを拾う）
    cand_targets = [f"{prefix}_tg", f"{prefix}_targets", f"{prefix}_対象物", f"{prefix}_selected_targets"]
    cand_types   = [f"{prefix}_tp", f"{prefix}_types",   f"{prefix}_研究タイプ", f"{prefix}_selected_types"]

    def pick(keys: list[str]) -> list[str]:
        for k in keys:
            v = st.session_state.get(k)
            if v:
                if isinstance(v, (list, tuple, set)):
                    return [str(x).strip() for x in v if str(x).strip()]
                if isinstance(v, str):
                    return _split_simple(v)
        return []

    tg_sel = pick(cand_targets)
    tp_sel = pick(cand_types)
    return y_from, y_to, tg_sel, tp_sel