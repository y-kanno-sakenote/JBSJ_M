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

    # 左に表、右にグラフ（研究者タブの論文数サブタブと同一レイアウトに合わせる）
    left, right = st.columns([1.0, 1.1])
    display_height = 420
    with left:
        # 表示用に上位 topn 件をそのまま表示
        try:
            st.dataframe(freq_df, use_container_width=True, hide_index=True, height=display_height)
        except Exception:
            st.dataframe(freq_df, use_container_width=True, hide_index=True)



    with right:
        if HAS_PX:
            # グラフは常に Top10 のみ表示（左の表は topn に従う）
            try:
                df_chart = freq_df.sort_values("件数", ascending=False).head(10)
                # 横棒にする：件数を x、キーワードを y にして orientation='h'
                df_plot = df_chart.sort_values("件数", ascending=True)
                fig = px.bar(df_plot, x="件数", y="キーワード", orientation='h', text_auto=True, title=f"頻出キーワード（Top10）{title_suffix}")
                # make bars visually thicker by reducing gap and removing border lines
                # use same top margin as coauthor charts to match vertical alignment
                fig.update_layout(margin=dict(l=6, r=6, t=40, b=6), height=display_height, bargap=0.20, bargroupgap=0.06)
                fig.update_yaxes(automargin=True)
                fig.update_traces(marker_line_width=0)
                # Remove default xaxis title and instead place '件数' as a right-aligned annotation below the axis
                fig.update_layout(xaxis_title="")
                # place '件数' slightly above/right of the x-axis tick labels (paper coords)
                fig.update_layout(annotations=[
                    dict(
                        x=1.0,
                        y=-0.02,
                        xref='paper',
                        yref='paper',
                        text='件数',
                        showarrow=False,
                        xanchor='right',
                        yanchor='bottom',
                    )
                ])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # フォールバック: 元の縦棒表示（Top10）
                df_chart = freq_df.sort_values("件数", ascending=False).head(10)
                fig = px.bar(df_chart, x="キーワード", y="件数", text_auto=True, title=f"頻出キーワード（Top10）{title_suffix}")
                fig.update_layout(margin=dict(l=6, r=6, t=40, b=6), height=display_height, bargap=0.20, bargroupgap=0.06)
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Plotly が無い場合は既存の streamlit 縦棒を表示
            # show top10 in fallback as well (st.bar_chart doesn't accept height)
            st.bar_chart(freq_df.set_index("キーワード")["件数"].sort_values(ascending=False).head(10))

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