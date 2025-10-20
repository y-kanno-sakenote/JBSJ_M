from __future__ import annotations
from typing import List
import pandas as pd
import streamlit as st
from .base import split_multi
from .stopwords import clean_token

KEY_COLS = ["llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
            "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
            "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10"]

def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    words: List[str] = []
    for c in KEY_COLS:
        if c in row and pd.notna(row[c]):
            for w in split_multi(row[c]):
                cw = clean_token(w)
                if cw: words.append(cw)
    return words

@st.cache_data(ttl=600, show_spinner=False)
def collect_keywords(df: pd.DataFrame) -> pd.Series:
    bags: List[str] = []
    for _, r in df.iterrows(): bags += _extract_keywords_from_row(r)
    return pd.Series(bags, dtype="object")

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq(df: pd.DataFrame) -> pd.Series:
    s = collect_keywords(df)
    if s.empty: return pd.Series(dtype=int)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq_by_mode(df: pd.DataFrame, mode: str = "df") -> pd.Series:
    if mode == "df":
        bags: list[str] = []
        for _, r in df.iterrows():
            kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
            bags.extend(kws)
        if not bags: return pd.Series(dtype=int)
        return pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)
    return keyword_freq(df)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): continue
        kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
        for k in kws: rows.append((int(y), k))
    if not rows: return pd.DataFrame(columns=["発行年", "keyword", "count"])
    c = pd.DataFrame(rows, columns=["発行年","keyword"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def build_keyword_cooccur_edges(df: pd.DataFrame, min_edge: int) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        kws = sorted(set(_extract_keywords_from_row(r)))
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                rows.append((kws[i], kws[j]))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges