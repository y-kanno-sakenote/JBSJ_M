# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析（高速版・キャッシュ付）
① 頻出キーワード（表 / バーチャート / 任意: ワードクラウド）
② 共起キーワードネットワーク（任意: networkx + pyvis）
③ トレンド分析（年別出現推移）

依存はすべてオプショナルにし、未導入でも機能が落ちないように実装。
"""

from __future__ import annotations
import re
import itertools
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd
import streamlit as st

# ---- Optional deps ----
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:
    HAS_WC = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False


# =========================================================
# 前処理・ユーティリティ（キャッシュ対応）
# =========================================================

KW_COLS_DEFAULT = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]

_SPLIT_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def _norm(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_lc(s: str) -> str:
    return _norm(s).lower()

def _split_tokens(val: str) -> list[str]:
    if val is None:
        return []
    return [w.strip() for w in _SPLIT_RE.split(str(val)) if w.strip()]

def _collect_tokens_from_row(row: pd.Series, kw_cols: list[str]) -> list[str]:
    toks: list[str] = []
    for c in kw_cols:
        if c in row and pd.notna(row[c]):
            toks.extend(_split_tokens(row[c]))
    # ノイズ除去 & 小文字化
    toks = [_norm_lc(t) for t in toks if t]
    # 1論文で同語が重複していてもOK（頻度は積み上げる）
    return toks

@st.cache_data(ttl=600, show_spinner=False)
def precompute_keywords_df(
    df: pd.DataFrame,
    kw_cols: tuple[str, ...]
) -> pd.DataFrame:
    """
    1度だけ全行をなめて、後段のフィルタに効く情報をキャッシュする。
    戻り値: DataFrame[発行年_num, targets_lc, types_lc, tokens(list[str])]
    """
    use = df.copy()

    # 年
    if "発行年" in use.columns:
        use["発行年_num"] = pd.to_numeric(use["発行年"], errors="coerce")
    else:
        use["発行年_num"] = pd.Series([None] * len(use), dtype="float")

    # 対象物/研究タイプ（文字列化・小文字化）
    tgt_col = "対象物_top3" if "対象物_top3" in use.columns else None
    typ_col = "研究タイプ_top3" if "研究タイプ_top3" in use.columns else None

    use["targets_lc"] = use[tgt_col].astype(str).apply(_norm_lc) if tgt_col else ""
    use["types_lc"]   = use[typ_col].astype(str).apply(_norm_lc) if typ_col else ""

    # トークン化済みキーワード
    tokens_all = []
    for _, row in use.iterrows():
        toks = _collect_tokens_from_row(row, list(kw_cols))
        tokens_all.append(toks)
    use["tokens"] = tokens_all

    return use[["発行年_num","targets_lc","types_lc","tokens"]].copy()

def _year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025

def _apply_filters(
    base: pd.DataFrame,
    y_from: int, y_to: int,
    targets_sel: list[str],
    types_sel: list[str]
) -> Iterable[list[str]]:
    """
    前計算済みDF(base)に対して、年・対象物・研究タイプ条件で行を絞り、
    各行の tokens(list[str]) を順に返すジェネレータ。
    """
    # 年
    m = (base["発行年_num"].isna()) | ((base["発行年_num"] >= y_from) & (base["発行年_num"] <= y_to))

    # 対象物
    if targets_sel:
        t_norm = [_norm_lc(x) for x in targets_sel]
        m &= base["targets_lc"].apply(lambda s: any(t in s for t in t_norm))

    # 研究タイプ
    if types_sel:
        tt_norm = [_norm_lc(x) for x in types_sel]
        m &= base["types_lc"].apply(lambda s: any(t in s for t in tt_norm))

    for toks in base.loc[m, "tokens"]:
        yield toks


# =========================================================
# ① 頻出キーワード
# =========================================================

def _render_freq_block(df: pd.DataFrame, key_ns: str = "kw_freq") -> None:
    st.markdown("### 🔤 頻出キーワード")
    ymin, ymax = _year_bounds(df)

    # ---- 条件UI（キーは重複防止で名前空間を付ける）
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key=f"{key_ns}_year")
    with c2:
        targets_q = st.text_input("対象物（部分一致・カンマ/空白区切り）", key=f"{key_ns}_tgt")
        targets_sel = [w.strip() for w in _SPLIT_RE.split(targets_q) if w.strip()]
    with c3:
        types_q = st.text_input("研究タイプ（部分一致・カンマ/空白区切り）", key=f"{key_ns}_typ")
        types_sel = [w.strip() for w in _SPLIT_RE.split(types_q) if w.strip()]

    c4, c5 = st.columns([1, 1])
    with c4:
        top_n = st.number_input("上位N", min_value=10, max_value=200, value=30, step=10, key=f"{key_ns}_topn")
    with c5:
        do_wc = st.toggle("ワードクラウドも出す（任意）", value=False, key=f"{key_ns}_wc")

    # ---- 前処理をキャッシュして取得
    kw_cols = tuple([c for c in KW_COLS_DEFAULT if c in df.columns])
    base = precompute_keywords_df(df, kw_cols=kw_cols)

    # ---- 集計（軽い）
    cnt = Counter()
    for toks in _apply_filters(base, y_from, y_to, targets_sel, types_sel):
        cnt.update(toks)

    if not cnt:
        st.info("条件に一致するキーワードが見つかりませんでした。")
        return

    items = cnt.most_common(int(top_n))
    freq_df = pd.DataFrame(items, columns=["keyword", "count"])

    # 表
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

    # バー
    st.bar_chart(freq_df.set_index("keyword")["count"])

    # 任意: ワードクラウド
    if do_wc:
        if HAS_WC and HAS_MPL:
            wc = WordCloud(width=1024, height=512, background_color="white",
                           font_path=None).generate_from_frequencies(dict(cnt))
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("WordCloud / matplotlib が未導入のため表示をスキップします。")


# =========================================================
# ② 共起キーワードネットワーク
# =========================================================

def _render_cooccurrence_block(df: pd.DataFrame, key_ns: str = "kw_cooc") -> None:
    st.markdown("### 🔗 共起キーワードネットワーク（論文内で一緒に出る語）")

    ymin, ymax = _year_bounds(df)

    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    with c0:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key=f"{key_ns}_year")
    with c1:
        targets_q = st.text_input("対象物（部分一致・カンマ/空白区切り）", key=f"{key_ns}_tgt")
        targets_sel = [w.strip() for w in _SPLIT_RE.split(targets_q) if w.strip()]
    with c2:
        types_q = st.text_input("研究タイプ（部分一致・カンマ/空白区切り）", key=f"{key_ns}_typ")
        types_sel = [w.strip() for w in _SPLIT_RE.split(types_q) if w.strip()]
    with c3:
        min_w = st.number_input("共起回数の下限 w≥", min_value=1, max_value=50, value=2, step=1, key=f"{key_ns}_minw")

    c4, c5 = st.columns([1, 1])
    with c4:
        max_kw_per_doc = st.number_input("1論文あたり最大語数", min_value=5, max_value=50, value=15, step=5, key=f"{key_ns}_maxk")
    with c5:
        show_top = st.number_input("上位エッジ数（表）", min_value=20, max_value=500, value=100, step=20, key=f"{key_ns}_topE")

    kw_cols = tuple([c for c in KW_COLS_DEFAULT if c in df.columns])
    base = precompute_keywords_df(df, kw_cols=kw_cols)

    # エッジ集計（キャッシュ不要：フィルタ依存のため軽く実行）
    pair_cnt = Counter()
    for toks in _apply_filters(base, y_from, y_to, targets_sel, types_sel):
        if not toks:
            continue
        toks_uni = list(dict.fromkeys(toks))[: int(max_kw_per_doc)]  # 重複除去 + 上限
        for s, t in itertools.combinations(sorted(toks_uni), 2):
            pair_cnt[(s, t)] += 1

    # 下限でカット
    edges = [(a, b, w) for (a, b), w in pair_cnt.items() if w >= int(min_w)]
    if not edges:
        st.info("条件に一致する共起エッジがありませんでした。")
        return

    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    edge_df = pd.DataFrame(edges[: int(show_top)], columns=["src", "dst", "weight"])
    st.dataframe(edge_df, use_container_width=True, hide_index=True)

    # ネットワーク（任意）
    with st.expander("🕸️ ネットワーク可視化（networkx / pyvis 任意）", expanded=False):
        if HAS_NX and HAS_PYVIS:
            draw = st.button("🌐 可視化する", key=f"{key_ns}_draw")
            if draw:
                # Graph 構築
                G = nx.Graph()
                for s, t, w in edges:
                    G.add_edge(s, t, weight=int(w))
                # PyVis
                net = Network(height="700px", width="100%", bgcolor="#fff", font_color="#222")
                net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110)
                for n in G.nodes():
                    net.add_node(n, label=n)
                for s, t, d in G.edges(data=True):
                    w = int(d.get("weight", 1))
                    net.add_edge(s, t, value=w, title=f"共起: {w}")

                # ブラウザ自動オープンを避け、埋め込み用HTMLを生成
                html = net.generate_html(notebook=False)
                st.components.v1.html(html, height=720, scrolling=True)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示します。")


# =========================================================
# ③ トレンド分析（年別）
# =========================================================

def _render_trend_block(df: pd.DataFrame, key_ns: str = "kw_trend") -> None:
    st.markdown("### 📈 トレンド分析（年別出現頻度）")

    ymin, ymax = _year_bounds(df)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(max(ymin, ymax-20), ymax), key=f"{key_ns}_year")
    with c2:
        targets_q = st.text_input("対象物（部分一致・カンマ/空白区切り）", key=f"{key_ns}_tgt")
        targets_sel = [w.strip() for w in _SPLIT_RE.split(targets_q) if w.strip()]
    with c3:
        types_q = st.text_input("研究タイプ（部分一致・カンマ/空白区切り）", key=f"{key_ns}_typ")
        types_sel = [w.strip() for w in _SPLIT_RE.split(types_q) if w.strip()]

    c4, c5 = st.columns([1, 1])
    with c4:
        top_n = st.number_input("Top N 語", min_value=5, max_value=50, value=15, step=5, key=f"{key_ns}_topn")
    with c5:
        min_year_hits = st.number_input("少なくとも出現する年数（ノイズ抑制）", min_value=1, max_value=10, value=2, step=1, key=f"{key_ns}_minyr")

    kw_cols = tuple([c for c in KW_COLS_DEFAULT if c in df.columns])
    base = precompute_keywords_df(df, kw_cols=kw_cols)

    # まずは対象範囲で頻出TopN語を選ぶ
    total_cnt = Counter()
    per_year = defaultdict(Counter)  # year -> Counter(keyword -> count)
    for toks in _apply_filters(base, y_from, y_to, targets_sel, types_sel):
        # 年の取得は base からできないので、再計算用に発行年を同じ条件で取る必要あり
        # → キャッシュDFの行スライスが分からないため、簡易に year を推定：Noneはスキップ
        #   ここでは precompute した DF を再度条件式で絞るのが最も正確
        pass

    # 正確に年別集計するには、適用対象の行インデックスにアクセスする必要があるので
    # ここだけは base を直接フィルタリングして明示的に回す
    m = (base["発行年_num"].isna()) | ((base["発行年_num"] >= y_from) & (base["発行年_num"] <= y_to))
    if targets_sel:
        t_norm = [_norm_lc(x) for x in targets_sel]
        m &= base["targets_lc"].apply(lambda s: any(t in s for t in t_norm))
    if types_sel:
        tt_norm = [_norm_lc(x) for x in types_sel]
        m &= base["types_lc"].apply(lambda s: any(t in s for t in tt_norm))

    sub = base.loc[m]
    if sub.empty:
        st.info("条件に一致するデータがありません。")
        return

    for _, r in sub.iterrows():
        year = r["発行年_num"]
        if pd.isna(year):
            continue
        year = int(year)
        toks = r["tokens"]
        total_cnt.update(toks)
        per_year[year].update(toks)

    # Top語抽出（少なくとも min_year_hits 年以上に出現）
    years_with_kw = Counter()
    for y, c in per_year.items():
        for k in c.keys():
            years_with_kw[k] += 1

    candidates = [k for k, v in total_cnt.most_common() if years_with_kw[k] >= int(min_year_hits)]
    top_keys = candidates[: int(top_n)]
    if not top_keys:
        st.info("条件に合う上位語がありませんでした。閾値を調整してください。")
        return

    # 折れ線用データ
    line_df = []
    for y in sorted(per_year.keys()):
        c = per_year[y]
        for k in top_keys:
            line_df.append({"year": y, "keyword": k, "count": c.get(k, 0)})
    line_df = pd.DataFrame(line_df)

    st.dataframe(
        line_df.pivot(index="year", columns="keyword", values="count").fillna(0).astype(int),
        use_container_width=True
    )

    # Streamlitのline_chartで簡潔に
    st.line_chart(
        line_df.pivot(index="year", columns="keyword", values="count").fillna(0),
        use_container_width=True
    )


# =========================================================
# エクスポートされる描画関数
# =========================================================

def render_keyword_tab(df: pd.DataFrame) -> None:
    """
    分析>キーワード タブのメイン。内部でサブセクションを段落化。
    それぞれのブロックは独立にキー空間を持ち、相互に干渉しない。
    """
    st.header("🧠 キーワード分析")

    with st.expander("① 頻出キーワード", expanded=True):
        _render_freq_block(df, key_ns="kw_freq")

    with st.expander("② 共起キーワードネットワーク", expanded=False):
        _render_cooccurrence_block(df, key_ns="kw_cooc")

    with st.expander("③ トレンド分析（年別推移）", expanded=False):
        _render_trend_block(df, key_ns="kw_trend")