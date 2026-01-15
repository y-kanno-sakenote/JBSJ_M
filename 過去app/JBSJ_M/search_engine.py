# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List
from normalization import normalize_query

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    # 年レンジ
    y1, y2 = filters.get("year_min"), filters.get("year_max")
    if y1 is not None:
        out = out[out["year"] >= y1]
    if y2 is not None:
        out = out[out["year"] <= y2]
    # 著者（部分一致; 正規化済み前提）
    authors = filters.get("authors") or []
    for a in authors:
        out = out[out["authors"].str.contains(a, case=False, na=False)]
    # タグ（AND/OR切替）
    tag_mode = filters.get("tag_mode","OR")
    tag_terms: List[str] = filters.get("tag_terms") or []
    if tag_terms:
        if tag_mode == "AND":
            for t in tag_terms:
                out = out[out["tags"].str.contains(t, case=False, na=False)]
        else:
            mask = False
            for t in tag_terms:
                mask = mask | out["tags"].str.contains(t, case=False, na=False)
            out = out[mask]
    return out

def keyword_search(df: pd.DataFrame, query: str, mode: str = "AND") -> pd.DataFrame:
    q = normalize_query(query)
    if not q:
        return df
    terms = [t for t in q.split() if t]
    if not terms:
        return df
    cols = ["title","authors","doi","url","pdf_url","tags"]
    mask = None
    for term in terms:
        m = False
        for c in cols:
            m = m | df[c].astype(str).str.contains(term, case=False, na=False)
        if mask is None:
            mask = m
        else:
            mask = (mask & m) if mode == "AND" else (mask | m)
    return df[mask] if mask is not None else df
