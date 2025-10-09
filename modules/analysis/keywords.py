# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析タブ（完成版・安全な遅延実行＆キャッシュ付き）

機能（従来どおり）:
① 頻出キーワード分析
   - 年・対象物・研究タイプで絞り込み
   - 出現回数上位をバーチャート表示
   - WordCloud（wordcloud があれば）を任意表示

② 共起キーワードネットワーク（重いので遅延描画）
   - 同一論文内のキーワード共起を networkx + pyvis で可視化
   - 「ネットワークを描画」ボタン押下時のみ生成
   - ディスクキャッシュ（modules/common/cache_utils.py）対応

③ トレンド分析（経年変化）
   - 年ごとに出現頻度を集計し、TopN語を折れ線で可視化（Plotlyがなければst.line_chart）

注意：
- import時に重い処理を一切走らせません（関数内のみで実行）
- ウィジェットkeyは "kw_*" 接頭で他タブと衝突しないようにしています
"""

from __future__ import annotations
import re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

from pathlib import Path

def _get_japanese_font_path() -> str | None:
    """日本語フォントのパスを返す。プロジェクト同梱を最優先。"""
    candidates = [
        "fonts/IPAexGothic.ttf",                            # ← 同梱推奨
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",      # mac
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

# ==== Optional deps（無くても動く） ====
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

try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network  # type: ignore
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# 永続キャッシュIO（あれば使う）
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========= ユーティリティ =========
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)

@st.cache_data(ttl=600, show_spinner=False)
def year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

def _apply_filters(df: pd.DataFrame,
                   y_from: int, y_to: int,
                   targets: List[str], types: List[str]) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    if targets and "対象物_top3" in use.columns:
        use = use[col_contains_any(use["対象物_top3"], targets)]
    if types and "研究タイプ_top3" in use.columns:
        use = use[col_contains_any(use["研究タイプ_top3"], types)]
    return use

def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    words: List[str] = []
    for c in KEY_COLS:
        if c in row and pd.notna(row[c]):
            words += split_multi(row[c])
    return [w for w in words if w]

@st.cache_data(ttl=600, show_spinner=False)
def collect_keywords(df: pd.DataFrame) -> pd.Series:
    """全行からキーワード列を抽出して1本のSeriesに"""
    bags: List[str] = []
    for _, r in df.iterrows():
        bags += _extract_keywords_from_row(r)
    return pd.Series(bags, dtype="object")

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq(df: pd.DataFrame) -> pd.Series:
    """キーワード頻度（降順）"""
    s = collect_keywords(df)
    if s.empty:
        return pd.Series(dtype=int)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    """年×語の件数（論文ごと重複除去）"""
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): 
            continue
        kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
        for k in kws:
            rows.append((int(y), k))
    if not rows:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    c = pd.DataFrame(rows, columns=["発行年","keyword"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

# ====== 共起エッジ（重い：キャッシュ対応） ======
@st.cache_data(ttl=600, show_spinner=False)
def build_keyword_cooccur_edges(df: pd.DataFrame, min_edge: int) -> pd.DataFrame:
    """
    同一論文内で共起する語のペアをカウント
    戻り値: ['src','dst','weight']
    """
    rows = []
    for _, r in df.iterrows():
        kws = sorted(set(_extract_keywords_from_row(r)))
        # 全組合せ
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                rows.append((kws[i], kws[j]))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def _freq_to_df(freq: pd.Series, topn: int) -> pd.DataFrame:
    if freq.empty:
        return pd.DataFrame(columns=["キーワード","件数"])
    df = freq.head(int(topn)).reset_index()
    df.columns = ["キーワード","件数"]
    return df

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("networkx / pyvis が未導入のため、表のみ表示しています。")
        return
    if edges.empty:
        st.warning("対象条件でエッジがありません。")
        return

    # 文字列IDに統一
    G = nx.Graph()
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"]); w = int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=120, spring_strength=0.02)
    net.from_nx(G)
    html = net.generate_html(notebook=False)  # ← ブラウザ自動オープン回避
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========= ① 頻出キーワード =========
def _render_freq_block(df: pd.DataFrame) -> None:
    st.markdown("### ① 頻出キーワード")

    ymin, ymax = year_min_max(df)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_freq_year")
    with c2:
        tg_txt = st.text_input("対象物で絞り込み（空白区切り・部分一致）", value="", key="kw_freq_tg")
        tg_needles = [w for w in _SPLIT_MULTI_RE.split(tg_txt) if w.strip()]
    with c3:
        tp_txt = st.text_input("研究タイプで絞り込み（空白区切り・部分一致）", value="", key="kw_freq_tp")
        tp_needles = [w for w in _SPLIT_MULTI_RE.split(tp_txt) if w.strip()]
    with c4:
        topn = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5, key="kw_freq_topn")

    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)
    freq = keyword_freq(use)
    freq_df = _freq_to_df(freq, int(topn))

    if freq_df.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    # バーチャート
    if HAS_PX:
        fig = pd.DataFrame(freq_df)  # 明示
        fig = px.bar(fig, x="キーワード", y="件数", text_auto=True, title="頻出キーワード（上位）")
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(freq_df.set_index("キーワード")["件数"])

    # WordCloud（任意）
    with st.expander("☁ WordCloud（任意）", expanded=False):
        if HAS_WC:
            if st.button("生成する", key="kw_wc_btn"):
                # 日本語フォントの解決
                font_path = _get_japanese_font_path()
                wc_kwargs = dict(
                    width=900, height=450, background_color="white",
                    prefer_horizontal=1.0, collocations=False
                )
                if font_path:
                    wc_kwargs["font_path"] = font_path
                else:
                    st.warning("日本語フォントが見つかりません。`fonts/IPAexGothic.ttf` を置くと文字化けしません。")

                # freq_df → dict に明示変換（型の揺れ対策）
                freq_dict = {str(row["キーワード"]): int(row["件数"]) for _, row in freq_df.iterrows()}

                # 生成
                wc = WordCloud(**wc_kwargs).generate_from_frequencies(freq_dict)

                # PIL画像として安全に表示（matplotlib不使用）
                import io
                buf = io.BytesIO()
                img = wc.to_image()
                img.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, use_container_width=True)
        else:
            st.caption("※ wordcloud が未導入のため非表示です。")
            
# ========= ② 共起ネットワーク（遅延描画） =========
def _render_cooccur_block(df: pd.DataFrame) -> None:
    st.markdown("### ② 共起キーワードネットワーク")

    ymin, ymax = year_min_max(df)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_co_year")
    with c2:
        min_edge = st.number_input("エッジ最小回数 (w≥)", min_value=1, max_value=50, value=3, step=1, key="kw_co_minw")
    with c3:
        topN = st.number_input("ノード上限（出現上位）", min_value=30, max_value=300, value=120, step=10, key="kw_co_topn")
    with c4:
        st.caption("重いので下のボタンで明示的に描画します。")

    c5, c6 = st.columns([1,1])
    with c5:
        tg_txt = st.text_input("対象物で絞り込み（空白区切り・部分一致）", value="", key="kw_co_tg")
        tg_needles = [w for w in _SPLIT_MULTI_RE.split(tg_txt) if w.strip()]
    with c6:
        tp_txt = st.text_input("研究タイプで絞り込み（空白区切り・部分一致）", value="", key="kw_co_tp")
        tp_needles = [w for w in _SPLIT_MULTI_RE.split(tp_txt) if w.strip()]

    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)

    # キャッシュキー
    cache_key = f"kwco|{y_from}-{y_to}|min{min_edge}|top{topN}|tg{','.join(tg_needles)}|tp{','.join(tp_needles)}"

    # 1) エッジ構築（重いので永続キャッシュ）
    edges = None
    if HAS_DISK_CACHE:
        path_edges = cache_csv_path("kw_co_edges", cache_key)
        cached = load_csv_if_exists(path_edges)
        if cached is not None:
            edges = cached

    if edges is None:
        edges = build_keyword_cooccur_edges(use, int(min_edge))
        # 上位ノードだけに制限
        if not edges.empty and int(topN) > 0:
            deg = pd.concat([edges.groupby("src")["weight"].sum(),
                             edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
            keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
            edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)
        if HAS_DISK_CACHE:
            save_csv(edges, path_edges)

    st.caption(f"エッジ数: {len(edges)}")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)

    with st.expander("🕸️ ネットワークを描画（PyVis / 任意依存）", expanded=False):
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="kw_co_draw"):
                _draw_pyvis_from_edges(edges, height_px=680)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")


# ========= ③ トレンド（経年変化） =========
def _render_trend_block(df: pd.DataFrame) -> None:
    st.markdown("### ③ トレンド分析（経年変化）")

    ymin, ymax = year_min_max(df)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_trend_year")
    with c2:
        topn = st.number_input("表示する語数（TopN）", min_value=5, max_value=50, value=15, step=5, key="kw_trend_topn")
    with c3:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="kw_trend_ma")

    use = _apply_filters(df, y_from, y_to, [], [])
    yearly = yearly_keyword_counts(use)
    if yearly.empty:
        st.info("データがありません。")
        return

    # 最新年付近のTopN語を選ぶ（全体上位だと凡例が多すぎるため）
    latest_year = yearly["発行年"].max()
    latest_top = (yearly[yearly["発行年"] == latest_year]
                  .sort_values("count", ascending=False)["keyword"]
                  .head(int(topn)).tolist())
    piv = (yearly[yearly["keyword"].isin(latest_top)]
           .pivot_table(index="発行年", columns="keyword", values="count", aggfunc="sum")
           .fillna(0).sort_index())

    if int(ma) > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    if HAS_PX:
        fig = px.line(piv.reset_index().melt(id_vars="発行年", var_name="キーワード", value_name="件数"),
                      x="発行年", y="件数", color="キーワード", markers=True)
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(piv)


# ========= エクスポート：タブ本体 =========
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧠 キーワード分析")

    tab1, tab2, tab3 = st.tabs([
        "① 頻出キーワード",
        "② 共起ネットワーク",
        "③ トレンド分析",
    ])

    with tab1:
        _render_freq_block(df)

    with tab2:
        _render_cooccur_block(df)   # ← 遅延描画（ボタン式）

    with tab3:
        _render_trend_block(df)