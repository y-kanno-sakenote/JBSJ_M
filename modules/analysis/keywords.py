# -*- coding: utf-8 -*-
"""
キーワード分析タブ
① 頻出キーワード（bar / wordcloud）
② 共起キーワードネットワーク（軽量化＋ディスクキャッシュ）
③ トレンド分析（経年変化）
"""

from __future__ import annotations
import re
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st

# ==== オプショナル依存 ====
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    from wordcloud import WordCloud  # pip install wordcloud
    HAS_WC = True
except Exception:
    HAS_WC = False

try:
    import networkx as nx  # pip install networkx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network  # pip install pyvis
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# ディスクキャッシュユーティリティ
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========= 正規化・分割 =========
_SPLIT_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def norm_key(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def split_multi(cell) -> List[str]:
    if not cell:
        return []
    return [w.strip() for w in _SPLIT_RE.split(str(cell)) if w.strip()]

def collect_keyword_columns(df: pd.DataFrame) -> List[str]:
    """ある可能性のあるキーワード列を、存在するものだけ返す"""
    candidates = [
        "primary_keywords","secondary_keywords","featured_keywords","llm_keywords",
        "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
        "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
        "keywords","keyword","KW","kw",
    ]
    cols = [c for c in candidates if c in df.columns]
    # 念のため追加で "キーワード" を前方一致で拾う
    cols += [c for c in df.columns if str(c).startswith("キーワード") and c not in cols]
    # 重複排除
    cols = list(dict.fromkeys(cols))
    return cols

def filter_df(df: pd.DataFrame,
              year_from: int|None, year_to: int|None,
              targets: List[str], types: List[str]) -> pd.DataFrame:
    out = df.copy()
    if year_from is not None and year_to is not None and "発行年" in out.columns:
        y = pd.to_numeric(out["発行年"], errors="coerce")
        out = out[(y >= year_from) & (y <= year_to) | y.isna()]
    if targets and "対象物_top3" in out.columns:
        tgt = [norm_key(t) for t in targets]
        out = out[out["対象物_top3"].fillna("").astype(str).map(lambda v: any(t in norm_key(v) for t in tgt))]
    if types and "研究タイプ_top3" in out.columns:
        tps = [norm_key(t) for t in types]
        out = out[out["研究タイプ_top3"].fillna("").astype(str).map(lambda v: any(t in norm_key(v) for t in tps))]
    return out


# ========= 頻度計算（キャッシュ） =========
@st.cache_data(ttl=600, show_spinner=False)
def _freq_count_cached(df: pd.DataFrame, year_from: int, year_to: int,
                       targets: List[str], types: List[str]) -> pd.DataFrame:
    use = filter_df(df, year_from, year_to, targets, types)
    kw_cols = collect_keyword_columns(use)
    if not kw_cols:
        return pd.DataFrame(columns=["keyword", "freq"])

    counter: Dict[str, int] = {}
    for _, row in use[kw_cols].fillna("").iterrows():
        tokens = []
        for c in kw_cols:
            tokens += split_multi(row[c])
        # 記述が重複していても純粋頻度でカウント（同一論文内で同語が複数列にあってもその分加算）
        for t in tokens:
            k = norm_key(t)
            if not k:
                continue
            counter[k] = counter.get(k, 0) + 1

    if not counter:
        return pd.DataFrame(columns=["keyword", "freq"])
    freq = pd.DataFrame(sorted(counter.items(), key=lambda x: (-x[1], x[0])), columns=["keyword", "freq"])
    return freq


# ========= 共起ネットワーク（軽量＋ディスクキャッシュ） =========
def _cooccur_cache_key(version: str,
                       year_from: int, year_to: int,
                       targets: List[str], types: List[str],
                       topK: int, min_w: int) -> str:
    # 文字列化して署名（cache_utils側でハッシュ化）
    return f"v{version}|y{year_from}-{year_to}|tg{tuple(sorted(targets))}|tp{tuple(sorted(types))}|K{topK}|w{min_w}"

@st.cache_data(ttl=600, show_spinner=False)
def _cooccur_edges_cached(df: pd.DataFrame, year_from: int, year_to: int,
                          targets: List[str], types: List[str],
                          topK: int, min_w: int,
                          use_disk_cache: bool) -> pd.DataFrame:
    """
    手順：
      1. フィルタ後の頻出 TopK を選定
      2. その語集合に限定して、同一行内共起をカウント
      3. min_w 以上のみ残す
      4. ディスクキャッシュ（CSV）にも保存/読込（任意）
    """
    version = "1"  # 将来フォーマット変えたら上げる
    sig = _cooccur_cache_key(version, year_from, year_to, targets, types, topK, min_w)

    # ディスクキャッシュ
    if use_disk_cache and HAS_DISK_CACHE:
        cache_path = cache_csv_path("kw_cooccur", sig)
        cached = load_csv_if_exists(cache_path)
        if cached is not None:
            return cached

    # 上位語を抽出
    freq = _freq_count_cached(df, year_from, year_to, targets, types)
    if freq.empty:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    vocab = set(freq.head(int(topK))["keyword"].tolist())

    use = filter_df(df, year_from, year_to, targets, types)
    kw_cols = collect_keyword_columns(use)
    if not kw_cols:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    pairs: Dict[Tuple[str, str], int] = {}
    for _, row in use[kw_cols].fillna("").iterrows():
        toks = []
        for c in kw_cols:
            toks += split_multi(row[c])
        toks = [norm_key(t) for t in toks if norm_key(t) in vocab]
        toks_uniq = sorted(set(toks))
        # 1文献内での重複は1回に（同語多出の偏りを抑える）
        for a, b in itertools.combinations(toks_uniq, 2):
            key = (a, b) if a <= b else (b, a)
            pairs[key] = pairs.get(key, 0) + 1

    if not pairs:
        edges = pd.DataFrame(columns=["src", "dst", "weight"])
    else:
        edges = pd.DataFrame(
            [(a, b, w) for (a, b), w in pairs.items() if w >= int(min_w)],
            columns=["src", "dst", "weight"]
        ).sort_values("weight", ascending=False).reset_index(drop=True)

    # ディスクキャッシュ保存
    if use_disk_cache and HAS_DISK_CACHE:
        try:
            save_csv(edges, cache_path)  # type: ignore[arg-type]
        except Exception:
            pass

    return edges


# ========= トレンド（時系列頻度） =========
@st.cache_data(ttl=600, show_spinner=False)
def _trend_timeseries_cached(df: pd.DataFrame,
                             y_from: int, y_to: int,
                             targets: List[str], types: List[str],
                             topN: int) -> pd.DataFrame:
    use = filter_df(df, y_from, y_to, targets, types)
    if "発行年" not in use.columns:
        return pd.DataFrame(columns=["year", "keyword", "freq"])

    kw_cols = collect_keyword_columns(use)
    if not kw_cols:
        return pd.DataFrame(columns=["year", "keyword", "freq"])

    # まず全体上位 topN を決めてから年別にカウント
    freq_all = _freq_count_cached(df, y_from, y_to, targets, types)
    use_vocab = set(freq_all.head(int(topN * 2))["keyword"].tolist())  # 少し余裕を持たせる

    rows = []
    y = pd.to_numeric(use["発行年"], errors="coerce")
    use = use.assign(_y=y)
    use = use[use["_y"].notna()].copy()
    for _, row in use[["_y"] + kw_cols].iterrows():
        yr = int(row["_y"])
        toks = []
        for c in kw_cols:
            toks += split_multi(row[c])
        toks = [norm_key(t) for t in toks if norm_key(t) in use_vocab]
        for t in toks:
            rows.append((yr, t))

    if not rows:
        return pd.DataFrame(columns=["year", "keyword", "freq"])

    ts = pd.DataFrame(rows, columns=["year", "keyword"])
    ts["freq"] = 1
    ts = (ts.groupby(["year", "keyword"])["freq"].sum()
                .reset_index()
                .sort_values(["keyword", "year"]))
    # 可視化で見やすいように keyword ごとに最大頻度が高い順で topN に絞る
    top_kw = (ts.groupby("keyword")["freq"].sum()
                .sort_values(ascending=False)
                .head(int(topN)).index.tolist())
    ts = ts[ts["keyword"].isin(top_kw)].copy()
    return ts


# ========= UI：頻出 =========
def _render_freq_block(df: pd.DataFrame):
    st.subheader("① 頻出キーワード")
    # 年レンジ
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_freq_year")
    with c2:
        topN = st.number_input("上位件数", min_value=10, max_value=200, value=50, step=10, key="kw_freq_topn")
    with c3:
        show_wc = st.toggle("ワードクラウドを表示", value=False, key="kw_freq_wc")

    c4, c5 = st.columns([1, 1])
    with c4:
        tg_raw = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tg_all = sorted(tg_raw)
        tg_sel = st.multiselect("対象物で絞り込み（部分一致）", tg_all, default=[], key="kw_freq_tg")
    with c5:
        tp_raw = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tp_all = sorted(tp_raw)
        tp_sel = st.multiselect("研究タイプで絞り込み（部分一致）", tp_all, default=[], key="kw_freq_tp")

    freq = _freq_count_cached(df, y_from, y_to, tg_sel, tp_sel)
    if freq.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    top = freq.head(int(topN)).copy()
    st.dataframe(top.rename(columns={"keyword": "キーワード", "freq": "出現数"}),
                 use_container_width=True, hide_index=True)

    if HAS_PX:
        fig = px.bar(top.head(30), x="keyword", y="freq",
                     labels={"keyword": "キーワード", "freq": "出現数"})
        fig.update_layout(xaxis_tickangle=-40, height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(top.set_index("keyword").head(30))

    if show_wc:
        if not HAS_WC:
            st.warning("wordcloud が未インストールです。`pip install wordcloud`")
        else:
            # 頻度辞書にして描画
            freqs = {r["keyword"]: int(r["freq"]) for _, r in top.iterrows()}
            wc = WordCloud(width=1000, height=400, background_color="white", font_path=None)
            img = wc.generate_from_frequencies(freqs).to_image()
            st.image(img, caption="Word Cloud", use_column_width=True)


# ========= UI：共起ネットワーク =========
def _render_cooccur_block(df: pd.DataFrame):
    st.subheader("② 共起キーワードネットワーク")

    # 年レンジ
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_net_year")
    with c2:
        topK = st.number_input("語彙の上位K（ノード数の上限）", min_value=30, max_value=500, value=150, step=10, key="kw_net_topk")
    with c3:
        min_w = st.number_input("最小共起回数 (w≥)", min_value=1, max_value=20, value=2, step=1, key="kw_net_minw")
    with c4:
        use_disk_cache = st.toggle("🗃 ディスクキャッシュを使う", value=True, key="kw_net_disk")

    c5, c6 = st.columns([1, 1])
    with c5:
        tg_raw = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tg_all = sorted(tg_raw)
        tg_sel = st.multiselect("対象物で絞り込み（部分一致）", tg_all, default=[], key="kw_net_tg")
    with c6:
        tp_raw = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tp_all = sorted(tp_raw)
        tp_sel = st.multiselect("研究タイプで絞り込み（部分一致）", tp_all, default=[], key="kw_net_tp")

    edges = _cooccur_edges_cached(df, y_from, y_to, tg_sel, tp_sel, int(topK), int(min_w), bool(use_disk_cache))

    if edges.empty:
        st.info("条件に合う共起関係が見つかりませんでした。パラメータを調整してください。")
        return

    st.caption(f"エッジ数: {len(edges)}（w≥{int(min_w)}）")
    st.dataframe(edges.head(50), use_container_width=True, hide_index=True)

    # PyVisで可視化（依存があれば）
    if HAS_NX and HAS_PYVIS:
        G = nx.Graph()
        for _, r in edges.iterrows():
            s, t, w = str(r["src"]), str(r["dst"]), int(r["weight"])
            G.add_edge(s, t, weight=w)

        net = Network(height="650px", width="100%", bgcolor="#fff", font_color="#222")
        net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110, spring_strength=0.02)
        net.from_nx(G)

        # 直接 show() はブラウザを開こうとするので、generate_html() 相当の流儀で埋め込み
        html_path = Path("kw_cooccur_network.html")
        net.write_html(str(html_path), open_browser=False, notebook=False)  # = generate & save
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=650, scrolling=True)
    else:
        st.info("可視化には networkx / pyvis が必要です。表のプレビューのみ表示しています。")


# ========= UI：トレンド =========
def _render_trend_block(df: pd.DataFrame):
    st.subheader("③ キーワード・トレンド（経年変化）")

    # 年レンジ
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw_trend_year")
    with c2:
        topN = st.number_input("上位語（可視化）", min_value=5, max_value=50, value=15, step=5, key="kw_trend_topn")
    with c3:
        smooth = st.toggle("移動平均（3年）で平滑化", value=False, key="kw_trend_smooth")

    c4, c5 = st.columns([1, 1])
    with c4:
        tg_raw = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tg_all = sorted(tg_raw)
        tg_sel = st.multiselect("対象物で絞り込み（部分一致）", tg_all, default=[], key="kw_trend_tg")
    with c5:
        tp_raw = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tp_all = sorted(tp_raw)
        tp_sel = st.multiselect("研究タイプで絞り込み（部分一致）", tp_all, default=[], key="kw_trend_tp")

    ts = _trend_timeseries_cached(df, y_from, y_to, tg_sel, tp_sel, int(topN))
    if ts.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    if smooth:
        # 3年移動平均
        ts = ts.sort_values(["keyword", "year"]).copy()
        ts["freq_smooth"] = ts.groupby("keyword")["freq"].transform(lambda s: s.rolling(3, 1).mean())
        y_col = "freq_smooth"
    else:
        y_col = "freq"

    if HAS_PX:
        fig = px.line(ts, x="year", y=y_col, color="keyword",
                      labels={"year": "年", y_col: "出現数（平滑化）" if smooth else "出現数"})
        fig.update_layout(legend_title_text="キーワード", height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(ts.pivot_table(index="year", columns="keyword", values=y_col, aggfunc="sum").sort_index())

    st.caption("※ トレンドは本文・要旨・メタデータに含まれるキーワード列の単純頻度を年別集計しています。")


# ========= エクスポート関数 =========
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧠 キーワード分析")
    sub1, sub2, sub3 = st.tabs(["① 頻出", "② 共起ネットワーク", "③ トレンド"])
    with sub1:
        _render_freq_block(df)
    with sub2:
        _render_cooccur_block(df)
    with sub3:
        _render_trend_block(df)