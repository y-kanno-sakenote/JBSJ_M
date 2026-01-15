# modules/analysis/targettype_mod/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple
import pandas as pd

TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

def _order_options(all_options: list[str], preferred: list[str]) -> list[str]:
    s = set(all_options)
    head = [x for x in preferred if x in s]
    tail = sorted([x for x in all_options if x not in preferred])
    return head + tail

_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

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

def year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

def apply_filters(df: pd.DataFrame, y_from: int, y_to: int, targets: List[str], types: List[str]) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    if targets and "対象物_top3" in use.columns:
        use = use[col_contains_any(use["対象物_top3"], targets)]
    if types and "研究タイプ_top3" in use.columns:
        use = use[col_contains_any(use["研究タイプ_top3"], types)]
    return use

def node_options_for_mode(df_use: pd.DataFrame, mode: str) -> list[str]:
    if mode == "対象物のみ":
        cand = sorted({t for v in df_use.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        return _order_options(cand, TARGET_ORDER)
    elif mode == "研究タイプのみ":
        cand = sorted({t for v in df_use.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        return _order_options(cand, TYPE_ORDER)
    else:
        cand_tg = sorted({t for v in df_use.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        cand_tp = sorted({t for v in df_use.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        cand_tg = _order_options(cand_tg, TARGET_ORDER)
        cand_tp = _order_options(cand_tp, TYPE_ORDER)
        return cand_tg + [x for x in cand_tp if x not in cand_tg]

def prefer_title_column(df: pd.DataFrame) -> str | None:
    for c in ["タイトル", "論文タイトル", "title", "Title", "題名"]:
        if c in df.columns:
            return c
    return None