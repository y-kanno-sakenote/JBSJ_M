# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from config import APP_TITLE
from data_loader import load_dataset
from search_engine import apply_filters, keyword_search
from ui_components import render_header, render_search_and_filters, render_results_table, render_favorites_editor
from favorites import filter_only_favorites

st.set_page_config(page_title=APP_TITLE, layout="wide")
render_header(APP_TITLE)

# データ読み込み
df = load_dataset()

# 検索・フィルタUI
params = render_search_and_filters(df)

# 検索処理
df_filtered = apply_filters(df, params)
df_searched = keyword_search(df_filtered, params["query"], mode=params["q_mode"])

st.subheader("検索結果")
render_results_table(df_searched, key_prefix="search")

# お気に入り：現在のフィルタ結果からお気に入りのみを抽出（全体から抽出に変えてもOK）
fav_df = filter_only_favorites(df)
fav_df = render_favorites_editor(fav_df)

# 保存などの処理（必要ならファイルに書き出し）
with st.expander("保存 / エクスポート", expanded=False):
    if st.button("お気に入りCSVを書き出す"):
        csv = fav_df.to_csv(index=False)
        st.download_button("Download favorites.csv", data=csv, file_name="favorites.csv", mime="text/csv")
