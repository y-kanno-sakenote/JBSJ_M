# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析タブ：
① 頻出キーワード分析（年/対象物/研究タイプでフィルタ → 上位を可視化）
② 共起キーワードネットワーク（同一論文内で共起する語をエッジ化）
③ トレンド分析（年ごとの出現頻度 ↗︎/↘︎ を可視化）

依存はすべて任意扱いにしています（無ければ自動フォールバック）：
- plotly（棒/折れ線）→ 無い場合は Streamlit の簡易チャート
- wordcloud（ワードクラウド）→ 無ければスキップ
- networkx & pyvis（共起ネットワーク）→ 無ければスキップ
"""

from __future__ import annotations

import re
import itertools
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Dict

import pandas as pd
import streamlit as st

# --- optional deps (無ければフォールバック) ---
try:
    import plotly.express as px  # type: ignore
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud  # type: ignore
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

try:
    import networkx as nx  # type: ignore
    from pyvis.network import Network  # type: ignore
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


# ========= ユーティリティ =========
_SPLIT_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def split_multi(cell) -> List[str]:
    """複数キーワードが区切りで入っているセルを分割"""
    if cell is None:
        return []
    return [w.strip() for w in _SPLIT_RE.split(str(cell)) if w.strip()]

def norm_token(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

@st.cache_data(ttl=600, show_spinner=False)
def collect_keywords(df: pd.DataFrame) -> List[str]:
    """
    DFのキーワード列を自動的に検出して、候補列名のリストを返す
    """
    cols = [str(c).strip() for c in df.columns]
    preferred = [
        "llm_keywords", "primary_keywords", "secondary_keywords", "featured_keywords",
        "keywords", "keyword"
    ]
    # キーワード1〜10系も拾う
    preferred += [c for c in cols if re.fullmatch(r"キーワード\d+", c)]
    # 既知の列だけ残す（順序保持）
    return [c for c in preferred if c in cols]

def concat_keywords_in_row(row: pd.Series, kw_cols: List[str]) -> List[str]:
    """1レコード内の複数列に散らばるキーワードをまとめてトークン配列に"""
    toks: List[str] = []
    for c in kw_cols:
        toks += split_multi(row.get(c, ""))
    # 重複除去（語形はそのまま、比較はlower）
    seen = set()
    out: List[str] = []
    for t in toks:
        k = norm_token(t)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t)   # “表記”は元のまま保持
    return out

def filter_df(df: pd.DataFrame,
              year_from: int, year_to: int,
              targets: list[str], types: list[str]) -> pd.DataFrame:
    """年・対象物・研究タイプでフィルタ"""
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

def make_year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025


# ========= ① 頻出キーワード分析 =========
def _render_freq_block(df: pd.DataFrame):
    st.markdown("### ① 頻出キーワード")

    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりませんでした。列名を確認してください。")
        return

    # フィルタUI
    ymin, ymax = make_year_min_max(df)
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[])
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[])

    use = filter_df(df, y_from, y_to, targets, types)

    # 集計
    counter = Counter()
    surface_form = {}  # lower -> 表記（最初に出たもの）
    for _, row in use.iterrows():
        toks = concat_keywords_in_row(row, kw_cols)
        for t in toks:
            k = norm_token(t)
            counter[k] += 1
            surface_form.setdefault(k, t)

    if not counter:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    topn = st.slider("上位表示件数", 5, 100, 30, 5)
    items = counter.most_common(topn)
    freq_df = pd.DataFrame([{"キーワード": surface_form[k], "出現数": n} for k, n in items])

    # 棒グラフ
    if HAS_PLOTLY:
        fig = px.bar(freq_df.sort_values("出現数", ascending=True),
                     x="出現数", y="キーワード", orientation="h",
                     title="頻出キーワード（上位）")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    # ワードクラウド（任意）
    with st.expander("🧩 ワードクラウド（任意）", expanded=False):
        if not HAS_WORDCLOUD:
            st.info("wordcloud が未導入のため表示しません。")
        else:
            wc = WordCloud(width=900, height=500, background_color="white",
                           colormap=None, prefer_horizontal=0.9,
                           regexp=r"[^ \n]+").generate_from_frequencies(
                {surface_form[k]: v for k, v in dict(counter).items()}
            )
            st.image(wc.to_array(), caption="WordCloud", use_column_width=True)


# ========= ② 共起キーワードネットワーク =========
def _render_cooccurrence_block(df: pd.DataFrame):
    st.markdown("### ② 共起キーワードネットワーク")

    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりませんでした。列名を確認してください。")
        return

    # フィルタUI
    ymin, ymax = make_year_min_max(df)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw2_year")
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[], key="kw2_target")
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[], key="kw2_type")
    with c4:
        min_w = st.number_input("表示する最小共起回数 (w≥)", 1, 20, 2)

    use = filter_df(df, y_from, y_to, targets, types)

    # エッジ抽出
    pair_counter: Dict[Tuple[str, str], int] = Counter()
    label_form: Dict[str, str] = {}  # lower -> original
    for _, row in use.iterrows():
        toks = concat_keywords_in_row(row, kw_cols)
        # lowerでユニーク化
        lower_unique = {}
        for t in toks:
            k = norm_token(t)
            if k not in lower_unique:
                lower_unique[k] = t
                label_form.setdefault(k, t)
        lows = sorted(lower_unique.keys())
        for a, b in itertools.combinations(lows, 2):
            pair_counter[(a, b)] += 1

    if not pair_counter:
        st.info("共起関係が見つかりませんでした。")
        return

    # 上位表示
    edges = [(label_form[a], label_form[b], w) for (a, b), w in pair_counter.items()]
    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"])
    st.dataframe(edges_df.sort_values("weight", ascending=False).head(50),
                 use_container_width=True, hide_index=True)

    # ネットワーク描画（任意）
    with st.expander("🌐 ネットワークを描画（任意：networkx / pyvis）", expanded=False):
        if not HAS_GRAPH:
            st.info("networkx / pyvis が未導入のため描画をスキップします。")
            return

        # PyVis描画
        G = nx.Graph()
        for s, t, w in edges:
            if w < int(min_w):
                continue
            if G.has_edge(s, t):
                G[s][t]["weight"] += int(w)
            else:
                G.add_edge(s, t, weight=int(w))

        if G.number_of_edges() == 0:
            st.warning("指定した最小共起回数以上のエッジがありません。")
            return

        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#222")
        net.barnes_hut(central_gravity=0.25, spring_length=110, spring_strength=0.02)
        for n in G.nodes():
            net.add_node(n, label=n)
        for s, t, d in G.edges(data=True):
            w = int(d.get("weight", 1))
            net.add_edge(s, t, value=w, title=f"共起回数: {w}")

        # ブラウザ自動オープンを避ける：HTML生成して埋め込み
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=700, scrolling=True)


# ========= ③ トレンド（経年変化） =========
def _render_trend_block(df: pd.DataFrame):
    st.markdown("### ③ トレンド分析（経年変化）")

    kw_cols = collect_keywords(df)
    if not kw_cols:
        st.info("キーワード列が見つかりませんでした。列名を確認してください。")
        return

    ymin, ymax = make_year_min_max(df)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="kw3_year")
    with c2:
        target_candidates = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets = st.multiselect("対象物", target_candidates, default=[], key="kw3_target")
    with c3:
        type_candidates = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types = st.multiselect("研究タイプ", type_candidates, default=[], key="kw3_type")
    with c4:
        top_n = st.number_input("上位N語（全期間頻度で抽出）", min_value=3, max_value=30, value=10, step=1)

    use = filter_df(df, y_from, y_to, targets, types)

    if "発行年" not in use.columns:
        st.info("発行年列が無いため、時系列を描画できません。")
        return

    # 全期間で上位N語（lowerで集計）
    total_counter = Counter()
    surface_form = {}
    for _, row in use.iterrows():
        toks = concat_keywords_in_row(row, kw_cols)
        for t in toks:
            k = norm_token(t)
            total_counter[k] += 1
            surface_form.setdefault(k, t)

    if not total_counter:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    top_keys = [k for k, _ in total_counter.most_common(int(top_n))]
    keep_labels = {k: surface_form[k] for k in top_keys}

    # 年×語でカウント
    rows = []
    yser = pd.to_numeric(use["発行年"], errors="coerce")
    for (_, r), y in zip(use.iterrows(), yser):
        if pd.isna(y):
            continue
        toks = [norm_token(t) for t in concat_keywords_in_row(r, kw_cols)]
        present = set(toks)
        for k in top_keys:
            if k in present:
                rows.append((int(y), keep_labels[k]))

    if not rows:
        st.info("選択範囲内で上位語の出現がありません。")
        return

    trend = pd.DataFrame(rows, columns=["year", "keyword"]).value_counts().reset_index()
    trend.columns = ["year", "keyword", "count"]

    if HAS_PLOTLY:
        fig = px.line(trend.sort_values(["keyword", "year"]),
                      x="year", y="count", color="keyword",
                      markers=True,
                      title="上位キーワードの年別出現頻度")
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(trend.sort_values(["keyword", "year"]),
                     use_container_width=True, hide_index=True)


# ========= エクスポート関数 =========
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧠 キーワード分析")

    info_cols = []
    if "発行年" in df.columns: info_cols.append("発行年")
    if "対象物_top3" in df.columns: info_cols.append("対象物_top3")
    if "研究タイプ_top3" in df.columns: info_cols.append("研究タイプ_top3")
    if info_cols:
        st.caption("利用列: " + " / ".join(info_cols))

    with st.expander("使い方", expanded=False):
        st.markdown(
            "- フィルタ（年・対象物・研究タイプ）で絞ってから、各分析の上位語やトレンドを確認します。\n"
            "- 共起ネットワークは **networkx + pyvis** が入っていればインタラクティブ表示されます。\n"
            "- Plotly が無い環境でも最低限の表は見られるようフォールバックします。"
        )

    # ① 頻出
    _render_freq_block(df)
    st.divider()

    # ② 共起ネットワーク
    _render_cooccurrence_block(df)
    st.divider()

    # ③ トレンド
    _render_trend_block(df)