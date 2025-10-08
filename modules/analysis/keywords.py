# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析モジュール（軽量・キャッシュ対応版）
- 頻出キーワード（棒グラフ / ワードクラウド）
- 共起ネットワーク（簡易版 / キャッシュ対応）
- トレンド分析（年次頻度）
"""

import re
import itertools
from pathlib import Path
import pandas as pd
import streamlit as st

# === オプション依存 ===
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:
    HAS_WC = False

try:
    import networkx as nx
    from pyvis.network import Network
    HAS_NX = True
except Exception:
    HAS_NX = False


# === 共通ユーティリティ ===
_AUTHOR_SPLIT = re.compile(r"[;；,、，/／|｜\s　]+")

def split_multi(s):
    if not s:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def extract_keywords(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if "keyword" in c.lower()]
    words = []
    for c in cols:
        words += [w for v in df[c].fillna("") for w in split_multi(v)]
    return words


# === キャッシュヘルパ ===
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(prefix, *params) -> Path:
    key = "_".join(str(p) for p in params)
    import hashlib
    sig = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{prefix}_{sig}.csv"

def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def _save_cache(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass


# === 頻出キーワード ===
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧠 キーワード分析")

    if df is None or df.empty:
        st.info("データが読み込まれていません。")
        return

    # === フィルタ設定 ===
    y = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
    ymin, ymax = int(y.min()), int(y.max())

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_year")
    with c2:
        tg_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        tg_sel = st.multiselect("対象物で絞り込み", tg_all, default=[], key="kw_target")
    with c3:
        tp_all = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        tp_sel = st.multiselect("研究タイプで絞り込み", tp_all, default=[], key="kw_type")

    # --- フィルタ適用 ---
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    if tg_sel:
        use = use[use["対象物_top3"].apply(lambda v: any(t in str(v) for t in tg_sel))]
    if tp_sel:
        use = use[use["研究タイプ_top3"].apply(lambda v: any(t in str(v) for t in tp_sel))]

    st.divider()

    # === ① 頻出キーワード分析 ===
    st.subheader("① 頻出キーワード")

    cache_path = _cache_path("freq", y_from, y_to, "-".join(tg_sel), "-".join(tp_sel))
    freq_df = _load_cache(cache_path)
    if freq_df is None:
        words = extract_keywords(use)
        freq = pd.Series(words).value_counts().reset_index()
        freq.columns = ["keyword", "count"]
        _save_cache(freq, cache_path)
        freq_df = freq

    top_n = st.slider("上位N件", 10, 100, 30, key="kw_topn")
    st.dataframe(freq_df.head(top_n), use_container_width=True, hide_index=True)

    if HAS_PX:
        fig = px.bar(freq_df.head(top_n), x="keyword", y="count", title="頻出キーワード", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(freq_df.head(top_n).set_index("keyword"))

    if HAS_WC and not freq_df.empty:
        st.markdown("#### ワードクラウド")
        wc = WordCloud(width=600, height=300, background_color="white",
                       font_path=None, colormap="viridis").generate_from_frequencies(
                           dict(zip(freq_df["keyword"], freq_df["count"]))
                       )
        st.image(wc.to_array())

    st.divider()

    # === ② 共起キーワードネットワーク ===
    st.subheader("② 共起キーワードネットワーク")

    net_cache_path = _cache_path("cooccur", y_from, y_to, "-".join(tg_sel), "-".join(tp_sel))
    net_df = _load_cache(net_cache_path)
    if net_df is None:
        pairs = []
        kw_cols = [c for c in use.columns if "keyword" in c.lower()]
        for _, row in use.iterrows():
            kws = [w for c in kw_cols for w in split_multi(row.get(c, ""))]
            for a, b in itertools.combinations(sorted(set(kws)), 2):
                pairs.append(tuple(sorted([a, b])))
        net_df = pd.DataFrame(pairs, columns=["src", "dst"])
        net_df = net_df.value_counts().reset_index(name="weight")
        _save_cache(net_df, net_cache_path)

    st.caption(f"共起ペア数: {len(net_df):,}")
    min_w = st.slider("最小共起回数", 1, int(net_df['weight'].max() if not net_df.empty else 1), 2, key="kw_minw")

    if HAS_NX and not net_df.empty:
        net_df = net_df[net_df["weight"] >= min_w]
        G = nx.Graph()
        for _, r in net_df.iterrows():
            G.add_edge(r["src"], r["dst"], weight=r["weight"])
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]
        keep_nodes = {n for n, _ in top_nodes}
        H = G.subgraph(keep_nodes)
        net = Network(height="550px", width="100%", bgcolor="#FFFFFF", font_color="#222222")
        net.from_nx(H)
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=550, scrolling=True)
    else:
        st.info("networkx または pyvis が未インストールのため、ネットワークを生成できません。")

    st.divider()

    # === ③ トレンド分析 ===
    st.subheader("③ トレンド分析（経年変化）")

    trend_cache_path = _cache_path("trend", "-".join(tg_sel), "-".join(tp_sel))
    trend_df = _load_cache(trend_cache_path)
    if trend_df is None:
        all_kws = []
        kw_cols = [c for c in use.columns if "keyword" in c.lower()]
        for _, row in use.iterrows():
            y = row.get("発行年", None)
            kws = [w for c in kw_cols for w in split_multi(row.get(c, ""))]
            for w in kws:
                all_kws.append((y, w))
        trend_df = pd.DataFrame(all_kws, columns=["year", "keyword"])
        trend_df = trend_df.groupby(["year", "keyword"]).size().reset_index(name="count")
        _save_cache(trend_df, trend_cache_path)

    top_kw = trend_df.groupby("keyword")["count"].sum().sort_values(ascending=False).head(10).index.tolist()
    trend_plot = trend_df[trend_df["keyword"].isin(top_kw)]

    if HAS_PX and not trend_plot.empty:
        fig2 = px.line(trend_plot, x="year", y="count", color="keyword", markers=True, title="キーワードの経年変化")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(trend_plot.pivot_table(index="year", columns="keyword", values="count", aggfunc="sum").fillna(0))