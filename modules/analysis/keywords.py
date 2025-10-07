# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析タブ：
① 頻出キーワード分析（年/対象物/研究タイプでフィルタ → 上位を可視化）
② 共起キーワードネットワーク（同一論文内で共起する語をエッジ化）
③ トレンド分析（年ごとの出現頻度 ↗︎/↘︎ を可視化）
"""

from __future__ import annotations
import re
import itertools
from collections import Counter
import pandas as pd
import streamlit as st

# --- optional deps ---
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

try:
    import networkx as nx
    from pyvis.network import Network
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


# ========= 共通ユーティリティ =========
_SPLIT_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def split_multi(cell):
    if cell is None:
        return []
    return [w.strip() for w in _SPLIT_RE.split(str(cell)) if w.strip()]

def norm_token(s):
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

@st.cache_data(ttl=600)
def collect_keywords(df: pd.DataFrame):
    cols = [str(c).strip() for c in df.columns]
    preferred = [c for c in cols if "keyword" in c.lower() or "キーワード" in c]
    return preferred

def concat_keywords_in_row(row: pd.Series, kw_cols):
    toks = []
    for c in kw_cols:
        toks += split_multi(row.get(c, ""))
    seen = set()
    out = []
    for t in toks:
        k = norm_token(t)
        if k not in seen and k:
            seen.add(k)
            out.append(t)
    return out

def filter_df(df, year_from, year_to, targets, types):
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]
    if targets and "対象物_top3" in use.columns:
        ts = [norm_token(t) for t in targets]
        use = use[use["対象物_top3"].astype(str).apply(lambda v: any(t in norm_token(v) for t in ts))]
    if types and "研究タイプ_top3" in use.columns:
        tp = [norm_token(t) for t in types]
        use = use[use["研究タイプ_top3"].astype(str).apply(lambda v: any(t in norm_token(v) for t in tp))]
    return use

def make_year_min_max(df):
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025


# ========= ① 頻出キーワード =========
def _render_freq_block(df):
    st.markdown("### ① 頻出キーワード")
    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = make_year_min_max(df)
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kwtab_freq_year")
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[], key="kwtab_freq_targets")
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[], key="kwtab_freq_types")

    use = filter_df(df, y_from, y_to, targets, types)
    counter = Counter()
    surface_form = {}
    for _, row in use.iterrows():
        toks = concat_keywords_in_row(row, kw_cols)
        for t in toks:
            k = norm_token(t)
            counter[k] += 1
            surface_form.setdefault(k, t)
    if not counter:
        st.info("該当キーワードなし。")
        return

    topn = st.slider("上位表示件数", 5, 100, 30, 5, key="kwtab_freq_topn")
    items = counter.most_common(topn)
    freq_df = pd.DataFrame([{"キーワード": surface_form[k], "出現数": n} for k, n in items])

    if HAS_PLOTLY:
        fig = px.bar(freq_df.sort_values("出現数", ascending=True),
                     x="出現数", y="キーワード", orientation="h", title="頻出キーワード")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)


# ========= ② 共起ネットワーク =========
def _render_cooccurrence_block(df):
    st.markdown("### ② 共起キーワードネットワーク")
    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = make_year_min_max(df)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kwtab_cooc_year")
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[], key="kwtab_cooc_targets")
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[], key="kwtab_cooc_types")
    with c4:
        min_w = st.number_input("表示する最小共起回数 (w≥)", 1, 20, 2, key="kwtab_cooc_minw")

    use = filter_df(df, y_from, y_to, targets, types)
    pairs = Counter()
    label_form = {}
    for _, row in use.iterrows():
        toks = concat_keywords_in_row(row, kw_cols)
        lower_unique = {norm_token(t): t for t in toks}
        lows = sorted(lower_unique.keys())
        for a, b in itertools.combinations(lows, 2):
            pairs[(a, b)] += 1
            label_form.setdefault(a, lower_unique[a])
            label_form.setdefault(b, lower_unique[b])

    if not pairs:
        st.info("共起関係なし。")
        return

    edges = [(label_form[a], label_form[b], w) for (a, b), w in pairs.items() if w >= min_w]
    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    st.dataframe(edges_df.sort_values("weight", ascending=False).head(50),
                 use_container_width=True, hide_index=True)

    with st.expander("🌐 ネットワークを描画（任意）", expanded=False):
        if not HAS_GRAPH:
            st.info("networkx / pyvis 未導入。")
            return
        G = nx.Graph()
        for s, t, w in edges:
            G.add_edge(s, t, weight=int(w))
        net = Network(height="700px", width="100%", bgcolor="#fff", font_color="#222")
        for n in G.nodes():
            net.add_node(n, label=n)
        for s, t, d in G.edges(data=True):
            w = int(d.get("weight", 1))
            net.add_edge(s, t, value=w, title=f"共起回数: {w}")
        html = net.generate_html(notebook=False)   # ← show() は使わず generate_html()
        st.components.v1.html(html, height=700, scrolling=True)


# ========= ③ トレンド =========
def _render_trend_block(df):
    st.markdown("### ③ トレンド分析（経年変化）")
    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりません。")
        return

    ymin, ymax = make_year_min_max(df)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kwtab_trend_year")
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[], key="kwtab_trend_targets")
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[], key="kwtab_trend_types")
    with c4:
        top_n = st.number_input("上位N語（全期間頻度で抽出）", 3, 30, 10, 1, key="kwtab_trend_topn")

    use = filter_df(df, y_from, y_to, targets, types)
    if "発行年" not in use.columns:
        st.info("発行年列がありません。")
        return

    counter = Counter()
    surface = {}
    for _, row in use.iterrows():
        for t in concat_keywords_in_row(row, kw_cols):
            k = norm_token(t)
            counter[k] += 1
            surface.setdefault(k, t)
    top_keys = [k for k, _ in counter.most_common(top_n)]
    rows = []
    for _, row in use.iterrows():
        y = pd.to_numeric(row.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        toks = [norm_token(t) for t in concat_keywords_in_row(row, kw_cols)]
        for k in top_keys:
            if k in toks:
                rows.append((int(y), surface[k]))
    if not rows:
        st.info("該当データなし。")
        return

    trend = pd.DataFrame(rows, columns=["year", "keyword"]).value_counts().reset_index()
    trend.columns = ["year", "keyword", "count"]
    if HAS_PLOTLY:
        fig = px.line(trend, x="year", y="count", color="keyword", markers=True,
                      title="上位キーワードの年別出現頻度")
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(trend, use_container_width=True, hide_index=True)


# ========= メイン関数 =========
def render_keyword_tab(df: pd.DataFrame):
    st.markdown("## 🧠 キーワード分析")

    _render_freq_block(df)
    st.divider()
    _render_cooccurrence_block(df)
    st.divider()
    _render_trend_block(df)