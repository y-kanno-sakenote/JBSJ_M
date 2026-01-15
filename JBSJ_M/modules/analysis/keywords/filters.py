from __future__ import annotations
import pandas as pd
import streamlit as st
from typing import Any

TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

_HAS_COMMON_FILTERS = False
_LAST_FILTER_META: dict[str, Any] = {}
try:
    from modules.common.filters import render_filter_bar as _rfb  # type: ignore
    _HAS_COMMON_FILTERS = True
except Exception:
    try:
        from ..common.filters import render_filter_bar as _rfb  # type: ignore
        _HAS_COMMON_FILTERS = True
    except Exception:
        _rfb = None  # type: ignore

def _selected_filters(prefix: str = "kw") -> tuple[list[str], list[str]]:
    """
    Try to recover current selections for 対象物 / 研究タイプ from:
      1) _LAST_FILTER_META (preferred)
      2) st.session_state (fallback; prefix-aware)
    Returns (targets, types). Empty lists when nothing explicit is selected.
    """
    def _as_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple, set)):
            return [str(v).strip() for v in x if str(v).strip()]
        if isinstance(x, str):
            import re as _re
            return [s.strip() for s in _re.split(r"[,;；、，/／|｜\s\u3000]+", x) if s.strip()]
        return []

    targets: list[str] = []
    types: list[str] = []

    # 1) From meta
    try:
        meta = _LAST_FILTER_META or {}
        for k in [
            "targets","targets_sel","selected_targets","targets_labels",
            f"{prefix}_targets", f"{prefix}_targets_sel", f"{prefix}_selected_targets",
            "対象物", f"{prefix}_対象物",
        ]:
            if k in meta and not targets:
                targets = _as_list(meta.get(k))
        for k in [
            "types","types_sel","selected_types","types_labels",
            f"{prefix}_types", f"{prefix}_types_sel", f"{prefix}_selected_types",
            "研究タイプ", f"{prefix}_研究タイプ",
        ]:
            if k in meta and not types:
                types = _as_list(meta.get(k))
    except Exception:
        pass

    # 2) Fallback from session_state
    if not targets or not types:
        try:
            ss = st.session_state
            if not targets:
                for k in [f"{prefix}_targets", f"{prefix}_対象物", f"{prefix}_tg", f"{prefix}_selected_targets", "targets", "対象物"]:
                    if k in ss:
                        vals = _as_list(ss.get(k)); 
                        if vals: targets = vals; break
            if not types:
                for k in [f"{prefix}_types", f"{prefix}_研究タイプ", f"{prefix}_tp", f"{prefix}_selected_types", "types", "研究タイプ"]:
                    if k in ss:
                        vals = _as_list(ss.get(k)); 
                        if vals: types = vals; break
        except Exception:
            pass

    return targets, types

def _df_from_result(res, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(res, pd.DataFrame): return res
    if isinstance(res, (list, tuple)) and res and isinstance(res[0], pd.DataFrame): return res[0]
    if isinstance(res, dict):
        for k in ("df","df_use","filtered_df"):
            v = res.get(k); 
            if isinstance(v, pd.DataFrame): return v
    return fallback_df

def safe_filter_bar(df: pd.DataFrame, key_prefix="kw", target_order=None, type_order=None) -> pd.DataFrame:
    global _LAST_FILTER_META
    if not _HAS_COMMON_FILTERS or _rfb is None:
        st.warning("共通フィルターの読込に失敗。データをそのまま使用します。", icon="⚠️")
        return df
    try:
        res = _rfb(df, key_prefix=key_prefix, target_order=target_order, type_order=type_order)
    except TypeError:
        try: res = _rfb(df, key_prefix=key_prefix)
        except TypeError:
            res = _rfb(df)
    except Exception as e:
        st.warning(f"共通フィルターで例外: {type(e).__name__}: {e}", icon="⚠️")
        return df
    _LAST_FILTER_META = res if isinstance(res, dict) else {}
    return _df_from_result(res, df)

def _fmt_list(name: str, vals: list[str] | None, max_items: int = 6):
    if not vals: return None
    vs = [str(v).strip() for v in vals if str(v).strip()]
    if not vs: return None
    txt = ", ".join(vs[:max_items]) + (" …" if len(vs) > max_items else "")
    return f"{name}：{txt}"

def render_provenance_banner_from_df(df_use: pd.DataFrame, total_n: int, y_from: int|None=None, y_to: int|None=None,
                                     tg_sel: list[str] | None=None, tp_sel: list[str] | None=None) -> None:
    try:
        # Auto-fill target/type when caller didn't pass them
        if not tg_sel and not tp_sel:
            _tg_auto, _tp_auto = _selected_filters(prefix="kw")
            if not tg_sel:
                tg_sel = _tg_auto
            if not tp_sel:
                tp_sel = _tp_auto
        n_filtered = len(df_use) if df_use is not None else 0
        if y_from is not None and y_to is not None:
            period = f"{int(y_from)}–{int(y_to)}"
        else:
            years = pd.to_numeric(df_use.get("発行年", pd.Series(dtype="object")), errors="coerce").dropna().astype(int) \
                    if (df_use is not None and "発行年" in df_use.columns) else pd.Series([], dtype=int)
            period = "—" if years.empty else f"{int(years.min())}–{int(years.max())}"
        parts = [f"出典：JBSJ DB（N={n_filtered} / {total_n}）", f"期間：{period}"]
        t1 = _fmt_list("対象物", tg_sel); t2 = _fmt_list("研究タイプ", tp_sel)
        if t1: parts.append(t1)
        if t2: parts.append(t2)
        st.caption(" ｜ ".join(parts))
    except Exception:
        st.caption(f"出典：JBSJ DB（N={len(df_use) if df_use is not None else 0} / {total_n}）")