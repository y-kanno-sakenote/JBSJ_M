# modules/analysis/targettype_mod/compute.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import itertools
from typing import List, Tuple
import pandas as pd
import streamlit as st
from .base import split_multi, prefer_title_column

@st.cache_data(ttl=600, show_spinner=False)
def count_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    bags: List[str] = []
    for v in df[col].fillna(""):
        bags += split_multi(v)
    if not bags:
        return pd.Series(dtype=int)
    s = pd.Series(bags)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def cross_counts(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    if col_a not in df.columns or col_b not in df.columns:
        return pd.DataFrame(columns=["A", "B", "count"])
    rows = []
    for _, r in df.iterrows():
        As = list(dict.fromkeys(split_multi(r.get(col_a, ""))))
        Bs = list(dict.fromkeys(split_multi(r.get(col_b, ""))))
        for a in As:
            for b in Bs:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", col, "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        items = list(dict.fromkeys(split_multi(r.get(col, ""))))
        for it in items:
            rows.append((int(y), it))
    if not rows:
        return pd.DataFrame(columns=["発行年", col, "count"])
    c = pd.DataFrame(rows, columns=["発行年", col]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def build_cooccur_edges(df: pd.DataFrame, mode: str, min_edge: int) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        tg = list(dict.fromkeys(split_multi(r.get("対象物_top3", ""))))
        tp = list(dict.fromkeys(split_multi(r.get("研究タイプ_top3", ""))))
        if mode == "対象物のみ":
            items = tg
            pairs = itertools.combinations(sorted(items), 2)
        elif mode == "研究タイプのみ":
            items = tp
            pairs = itertools.combinations(sorted(items), 2)
        else:
            pairs = itertools.product(sorted(set(tg)), sorted(set(tp)))
        for a, b in pairs:
            if a and b and a != b:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def ordered_index_and_columns(piv: pd.DataFrame, target_order: list[str], type_order: list[str]) -> tuple[list[str], list[str]]:
    cols = list(piv.columns)
    idxs  = list(piv.index)
    cols_order = [x for x in target_order if x in cols] + sorted([x for x in cols if x not in target_order])
    idx_order  = [x for x in type_order   if x in idxs] + sorted([x for x in idxs if x not in type_order])
    return idx_order, cols_order

def example_titles_for_edge(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    title_col = prefer_title_column(df)
    if title_col is None:
        return []
    tg_col = "対象物_top3"; tp_col = "研究タイプ_top3"
    def _tokset(val: str) -> set[str]:
        return set(split_multi(val))
    rows = []
    for _, r in df.iterrows():
        tg = _tokset(r.get(tg_col, "")) if tg_col in df.columns else set()
        tp = _tokset(r.get(tp_col, "")) if tp_col in df.columns else set()
        ok = (a in tg and b in tg) if mode == "対象物のみ" else ((a in tp and b in tp) if mode == "研究タイプのみ" else (a in tg and b in tp))
        if ok:
            rows.append((r.get("発行年", None), str(r.get(title_col, ""))))
    if not rows:
        return []
    try:
        rows = sorted(rows, key=lambda x: (pd.to_numeric(x[0], errors="coerce") if x[0] is not None else -1), reverse=True)
    except Exception:
        pass
    return [t for _, t in rows[:limit]]