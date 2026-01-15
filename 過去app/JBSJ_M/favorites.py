# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

FAV_KEY = "favorites_set"

def _ensure_state():
    if FAV_KEY not in st.session_state:
        st.session_state[FAV_KEY] = set()

def get_favorites() -> set[str]:
    _ensure_state()
    return st.session_state[FAV_KEY]

def toggle_favorite(item_id: str):
    _ensure_state()
    favs = st.session_state[FAV_KEY]
    if item_id in favs:
        favs.remove(item_id)
    else:
        favs.add(item_id)

def mark_favorites_column(df: pd.DataFrame) -> pd.DataFrame:
    favs = get_favorites()
    df = df.copy()
    df["★"] = df["id"].apply(lambda x: "★" if x in favs else "☆")
    return df

def filter_only_favorites(df: pd.DataFrame) -> pd.DataFrame:
    favs = get_favorites()
    return df[df["id"].isin(favs)]
