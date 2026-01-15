# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, List
from favorites import toggle_favorite, mark_favorites_column
from config import DEFAULT_PAGE_SIZE

def render_header(title: str):
    st.title(title)

def render_search_and_filters(df: pd.DataFrame) -> Dict[str, Any]:
    with st.expander("🔎 検索とフィルタ", expanded=True):
        col1, col2 = st.columns([2,1])
        with col1:
            query = st.text_input("キーワード（空白区切り / AND/ORは下で選択）", value="")
        with col2:
            q_mode = st.radio("検索モード", ["AND","OR"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            year_min = st.number_input("発行年（最小）", value=int(df["year"].min()) if not df.empty else 1900)
        with c2:
            year_max = st.number_input("発行年（最大）", value=int(df["year"].max()) if not df.empty else 2100)
        with c3:
            author_q = st.text_input("著者（部分一致 / カンマ区切り）", value="")
        t1, t2 = st.columns([3,1])
        with t1:
            tag_q = st.text_input("タグ検索（カンマ区切り）", value="")
        with t2:
            tag_mode = st.selectbox("タグ条件", ["OR","AND"], index=0)
    authors = [a.strip() for a in author_q.split(",") if a.strip()]
    tag_terms = [t.strip() for t in tag_q.split(",") if t.strip()]
    return {
        "query": query, "q_mode": q_mode,
        "year_min": year_min, "year_max": year_max,
        "authors": authors,
        "tag_terms": tag_terms, "tag_mode": tag_mode
    }

def render_results_table(df: pd.DataFrame, key_prefix: str = "results"):
    if df.empty:
        st.info("該当なし")
        return
    df_show = df.copy()
    df_show = mark_favorites_column(df_show)
    # 列順
    cols = ["★","title","authors","year","doi","url","pdf_url","tags","id"]
    cols = [c for c in cols if c in df_show.columns]
    df_show = df_show[cols]
    st.caption("クリックでお気に入り切替 / タグは後段の『お気に入り』で編集可")
    # インタラクション：各行のスターをクリックするUI
    for _, row in df_show.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([12,1])
            with c1:
                st.markdown(f"**{row.get('title','')}**")
                st.write(f"{row.get('authors','')}｜{row.get('year','')}")
                links = []
                if row.get("doi"):
                    links.append(f"[DOI]({row['doi']})")
                if row.get("url"):
                    links.append(f"[HP]({row['url']})")
                if row.get("pdf_url"):
                    links.append(f"[PDF]({row['pdf_url']})")
                if links:
                    st.markdown(" / ".join(links))
                if row.get("tags"):
                    st.caption(f"tags: {row['tags']}")
            with c2:
                if st.button(row["★"], key=f"{key_prefix}-fav-{row['id']}"):
                    toggle_favorite(row["id"])
    st.divider()

def render_favorites_editor(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    fav_ids = set(df["id"].unique())
    st.subheader("★ お気に入り（タグは直接編集可）")
    if df.empty:
        st.info("まだお気に入りがありません。上の検索結果で『☆』をクリックすると追加されます。")
        return df
    editable = df.copy()
    # 表示列の整頓
    show_cols = ["id","title","authors","year","tags","doi","url"]
    show_cols = [c for c in show_cols if c in editable.columns]
    editable = editable[show_cols]
    edited = st.data_editor(
        editable,
        num_rows="fixed",
        hide_index=True,
        key="fav_editor"
    )
    # 変更の反映（idをキーにtagsのみ上書き）
    merged = df.merge(edited[["id","tags"]], on="id", how="left", suffixes=("","_edit"))
    merged["tags"] = merged["tags_edit"].fillna(merged["tags"])
    merged = merged.drop(columns=["tags_edit"])
    return merged
