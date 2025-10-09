# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析タブ（軽量化対応版）
- ① 頻出キーワード分析：年・対象物・研究タイプで絞り込み、上位を棒グラフと（任意）ワードクラウド
- ② 共起キーワードネットワーク：同一論文内共起 → まず表でプレビュー、描画は「ボタンを押した時だけ」PyVis表示
- ③ トレンド分析：年毎の出現頻度を可視化（Plotlyが無ければst版にフォールバック）

※ 既存仕様は維持。共起ネットワークの“自動描画”のみ、Expander+ボタンに変更。
"""

from __future__ import annotations
import re
import itertools
from typing import List, Tuple, Dict, Iterable

import pandas as pd
import streamlit as st

# ---- Optional deps ----
try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

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

# ---- 永続キャッシュ（任意）----
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========= ユーティリティ =========
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

DEFAULT_KEYWORD_COLS = [
    "llm_keywords", "primary_keywords", "secondary_keywords", "featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]

def split_multi(s) -> List[str]:
    if not s: return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    if not needles:
        return pd.Series([True]*len(df_col), index=df_col.index)
    lo = [norm_key(x) for x in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo)
    return df_col.fillna("").astype(str).map(_hit)

@st.cache_data(ttl=600, show_spinner=False)
def _year_min_max(df: pd.DataFrame) -> tuple[int,int]:
    if "発行年" not in df.columns: return (1980, 2025)
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

def _available_keyword_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in DEFAULT_KEYWORD_COLS if c in df.columns]
    # 念のため、"keyword" を含む列も追加（重複排除）
    cols += [c for c in df.columns if ("keyword" in str(c).lower() and c not in cols)]
    return cols

def _iter_keywords_row(row: pd.Series, kw_cols: List[str]) -> Iterable[str]:
    for c in kw_cols:
        for w in split_multi(row.get(c, "")):
            yield w

# ========= ① 頻出キーワード =========
@st.cache_data(ttl=600, show_spinner=False)
def _keyword_freq(df: pd.DataFrame, kw_cols: List[str]) -> pd.Series:
    bag: List[str] = []
    for _, r in df.iterrows():
        bag += list(_iter_keywords_row(r, kw_cols))
    if not bag:
        return pd.Series(dtype=int)
    s = pd.Series(bag, dtype="object")
    return s.value_counts().sort_values(ascending=False)

def _render_freq_block(df: pd.DataFrame) -> None:
    st.markdown("### ① 頻出キーワード")
    kw_cols = _available_keyword_cols(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = _year_min_max(df)
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_freq_year")
    with c2:
        tg_sel = st.text_input("対象物フィルタ（任意・空白区切り）", value="", key="kw_freq_tg")
        tg_needles = [w for w in _SPLIT_MULTI_RE.split(tg_sel) if w.strip()]
    with c3:
        tp_sel = st.text_input("研究タイプフィルタ（任意・空白区切り）", value="", key="kw_freq_tp")
        tp_needles = [w for w in _SPLIT_MULTI_RE.split(tp_sel) if w.strip()]

    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)
    freq = _keyword_freq(use, kw_cols)

    if freq.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    top_n = st.number_input("表示件数", min_value=10, max_value=200, value=30, step=10, key="kw_freq_topn")
    freq_top = freq.head(int(top_n))

# --- ここから差し替え：_render_freq_block 内のバー描画パート丸ごと ---
    def _freq_to_df(freq_top):
        """Series / DataFrame いずれの形でも 'キーワード','件数' を揃える安全変換"""
        import pandas as pd
        if isinstance(freq_top, pd.Series):
            df = freq_top.rename("件数").reset_index().rename(columns={"index": "キーワード"})
        else:
            df = freq_top.reset_index()
            # 件数列の推定
            if "件数" not in df.columns:
                # index 以外の最初の列を件数扱い（v.value_counts() 由来を想定）
                cand = [c for c in df.columns if c != "index"]
                if cand:
                    df = df.rename(columns={cand[0]: "件数"})
            # キーワード列の統一
            if "キーワード" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "キーワード"})
        if "件数" in df.columns:
            df["件数"] = pd.to_numeric(df["件数"], errors="coerce").fillna(0).astype(int)
        return df

    # ここで freq_top を DataFrame 化してから描画
    freq_df = _freq_to_df(freq_top)

    if freq_df.empty:
        st.info("表示できるキーワードがありません。条件を見直してください。")
    else:
        try:
            import plotly.express as px
            fig = px.bar(
                freq_df,
                x="キーワード",
                y="件数",
                text_auto=True,
                title="頻出キーワード（上位）",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # Plotly が無い/エラー時はフォールバック
            st.bar_chart(freq_df.set_index("キーワード")["件数"])

    with st.expander("🧩 ワードクラウド（オプション）", expanded=False):
        try:
            from wordcloud import WordCloud  # type: ignore
            words = {k:int(v) for k,v in freq_top.to_dict().items()}
            if st.button("☁️ 生成する", key="kw_wc_btn"):
                wc = WordCloud(width=900, height=400, background_color="white",
                               font_path=None).generate_from_frequencies(words)
                st.image(wc.to_array(), use_column_width=True)
        except Exception:
            st.caption("wordcloud が未導入のためスキップしました。")


# ========= ② 共起キーワードネットワーク =========
@st.cache_data(ttl=600, show_spinner=False)
def _build_keyword_cooccur_edges(df: pd.DataFrame, kw_cols: List[str],
                                 min_edge: int) -> pd.DataFrame:
    """
    同一論文内で一緒に出たキーワード同士を1カウント。
    戻り値: ['src','dst','weight']
    """
    rows: List[Tuple[str,str]] = []
    for _, r in df.iterrows():
        terms = list(dict.fromkeys([w for w in _iter_keywords_row(r, kw_cols)]))
        if len(terms) < 2:
            continue
        for a, b in itertools.combinations(sorted(terms), 2):
            rows.append((a,b))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = (pd.DataFrame(rows, columns=["src","dst"])
             .value_counts()
             .reset_index(name="weight"))
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 680) -> None:
    """PyVisのHTML生成を使って埋め込む（ブラウザ自動起動を回避）"""
    if not (HAS_PYVIS and HAS_NX):
        st.info("networkx / pyvis が未導入のため、描画できません。")
        return
    if edges.empty:
        st.warning("描画対象のエッジがありません。")
        return

    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], int(r["weight"])
        if G.has_edge(s,t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s,t,weight=w)

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=120, spring_strength=0.02)
    net.from_nx(G)
    html = net.generate_html(notebook=False)  # ← ここがポイント（open_browserしない）
    st.components.v1.html(html, height=height_px, scrolling=True)

def _render_cooccur_block(df: pd.DataFrame) -> None:
    st.markdown("### ② 共起キーワードネットワーク（軽量表示）")
    kw_cols = _available_keyword_cols(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = _year_min_max(df)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_net_year")
    with c2:
        min_edge = st.number_input("エッジ最小回数 (w≥)", min_value=1, max_value=50, value=3, step=1, key="kw_net_minw")
    with c3:
        tg_sel = st.text_input("対象物フィルタ（任意）", value="", key="kw_net_tg")
        tg_needles = [w for w in _SPLIT_MULTI_RE.split(tg_sel) if w.strip()]
    with c4:
        tp_sel = st.text_input("研究タイプフィルタ（任意）", value="", key="kw_net_tp")
        tp_needles = [w for w in _SPLIT_MULTI_RE.split(tp_sel) if w.strip()]

    # 軽量フィルタを適用
    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)

    # ---- 永続キャッシュキー ----
    cache_key = f"kwco|{y_from}-{y_to}|min{min_edge}|tg{','.join(tg_needles)}|tp{','.join(tp_needles)}"

    # エッジ構築（重いのでディスクキャッシュ優先）
    edges = None
    if HAS_DISK_CACHE:
        p = cache_csv_path("kw_co_edges", cache_key)
        cached = load_csv_if_exists(p)
        if cached is not None:
            edges = cached

    if edges is None:
        edges = _build_keyword_cooccur_edges(use, kw_cols, int(min_edge))
        if HAS_DISK_CACHE:
            save_csv(edges, cache_csv_path("kw_co_edges", cache_key))

    st.caption(f"エッジ数: {len(edges)}（先頭200件をプレビュー）")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)

    # ==== ここが変更点：描画はユーザー操作時のみ ====
    with st.expander("🕸️ ネットワークを描画（任意・重い処理）", expanded=False):
        st.caption("ボタンを押すとPyVisでインタラクティブ描画します。")
        if st.button("🌐 ネットワークを描画する", key="kw_net_draw"):
            _draw_pyvis_from_edges(edges, height_px=680)


# ========= ③ トレンド（経年） =========
@st.cache_data(ttl=600, show_spinner=False)
def _yearly_keyword_counts(df: pd.DataFrame, kw_cols: List[str]) -> pd.DataFrame:
    """年×キーワードの件数（論文単位で重複排除）"""
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年","キーワード","count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): 
            continue
        items = list(dict.fromkeys([w for w in _iter_keywords_row(r, kw_cols)]))
        for it in items:
            rows.append((int(y), it))
    if not rows:
        return pd.DataFrame(columns=["発行年","キーワード","count"])
    c = (pd.DataFrame(rows, columns=["発行年","キーワード"])
         .value_counts()
         .reset_index(name="count"))
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

def _render_trend_block(df: pd.DataFrame) -> None:
    st.markdown("### ③ トレンド（経年変化）")
    kw_cols = _available_keyword_cols(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = _year_min_max(df)
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_trend_year")
    with c2:
        tg_sel = st.text_input("対象物フィルタ（任意）", value="", key="kw_trend_tg")
        tg_needles = [w for w in _SPLIT_MULTI_RE.split(tg_sel) if w.strip()]
    with c3:
        tp_sel = st.text_input("研究タイプフィルタ（任意）", value="", key="kw_trend_tp")
        tp_needles = [w for w in _SPLIT_MULTI_RE.split(tp_sel) if w.strip()]

    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)
    yearly = _yearly_keyword_counts(use, kw_cols)
    if yearly.empty:
        st.info("データがありません。")
        return

    all_terms = yearly["キーワード"].value_counts().head(500).index.tolist()
    sel = st.multiselect("表示するキーワード", all_terms[:1000], default=all_terms[: min(8, len(all_terms))], key="kw_trend_sel")

    piv = (yearly[yearly["キーワード"].isin(sel)]
           .pivot_table(index="発行年", columns="キーワード", values="count", aggfunc="sum")
           .fillna(0).sort_index())

    if piv.empty:
        st.info("選択語の系列が空です。")
        return

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
        "③ トレンド",
    ])
    with tab1:
        _render_freq_block(df)
    with tab2:
        _render_cooccur_block(df)   # ← 描画はボタン押下時のみ
    with tab3:
        _render_trend_block(df)