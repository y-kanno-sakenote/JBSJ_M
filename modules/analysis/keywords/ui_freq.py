from __future__ import annotations
import pandas as pd
import streamlit as st

try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    from wordcloud import WordCloud  # type: ignore
    HAS_WC = True
except Exception:
    HAS_WC = False

from .compute import keyword_freq_by_mode
from .images import get_japanese_font_path, safe_show_image
from .base import short_preview, get_banner_filters
from .copyui import expander as copy_expander

def _freq_to_df(freq: pd.Series, topn: int) -> pd.DataFrame:
    if freq.empty: return pd.DataFrame(columns=["キーワード","件数"])
    df = freq.head(int(topn)).reset_index()
    df.columns = ["キーワード","件数"]
    return df

def _build_caption(df_use: pd.DataFrame, topn: int, min_total: int, mode: str) -> str:
    y_from, y_to, tg_sel, tp_sel = get_banner_filters(prefix="kw")
    if y_from is not None and y_to is not None:
        period = f"{int(y_from)}–{int(y_to)}"
    else:
        period = "—"

    parts = [
        f"条件：表示件数：{int(topn)}",
        f"最低回数≧{int(min_total)}",
        "DF（登場論文数）" if mode=="df" else "TF（総出現回数）",
        f"期間：{period}",
    ]
    tg = short_preview(tg_sel or [])
    tp = short_preview(tp_sel or [])
    if tg:
        parts.append(f"対象物：{tg}")
    if tp:
        parts.append(f"研究タイプ：{tp}")
    return " ｜ ".join(parts)

def render_freq_block(df_use: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 1, 1.6])
    with c1:
        topn = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5, key="kw_freq_topn")
    with c2:
        min_total = st.number_input("最低総出現回数", min_value=1, max_value=100, value=3, step=1, key="kw_freq_min_total")
    with c3:
        label = st.radio("カウント方式", ["登場論文数（DF）", "総出現回数（TF）"], index=0, horizontal=True, key="kw_freq_countmode")
        mode = "df" if "DF" in label else "tf"

    freq = keyword_freq_by_mode(df_use, mode=mode)
    if freq.empty:
        st.info("条件に合うキーワードが見つかりませんでした。"); return
    if int(min_total) > 1:
        freq = freq[freq >= int(min_total)]

    freq_df = _freq_to_df(freq, int(topn))
    if freq_df.empty:
        st.info("（フィルタで該当なし）条件を緩めてください。"); return

    title_suffix = "（登場論文数）" if mode == "df" else "（出現回数）"
    if HAS_PX:
        fig = px.bar(freq_df, x="キーワード", y="件数", text_auto=True, title=f"頻出キーワード{title_suffix}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(freq_df.set_index("キーワード")["件数"])

    st.caption(_build_caption(df_use, topn, min_total, mode))
    copy_expander("📋 キーワードをすぐコピー", freq_df["キーワード"].astype(str).tolist())

    with st.expander("☁ WordCloud", expanded=False):
        if HAS_WC and st.button("生成する", key="kw_wc_btn"):
            textfreq = {row["キーワード"]: int(row["件数"]) for _, row in freq_df.iterrows()}
            wc = WordCloud(width=900, height=450, background_color="white",
                           collocations=False, prefer_horizontal=1.0,
                           font_path=get_japanese_font_path() or None)
            img = wc.generate_from_frequencies(textfreq).to_image()
            safe_show_image(img)
        elif not HAS_WC:
            st.caption("※ wordcloud が未導入のため非表示です。")