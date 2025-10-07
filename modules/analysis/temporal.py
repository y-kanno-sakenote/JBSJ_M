# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
経年変化（中心性スコアの推移）
- 年レンジ・対象物・研究タイプでフィルタ
- ウインドウ幅とステップで期間をスライド → 中心性（degree / betweenness / eigenvector）を算出
- 上位研究者のスコア推移を折れ線で可視化（Plotly が無ければ st.line_chart にフォールバック）
"""

from __future__ import annotations
import re
import itertools
from typing import Iterable, List, Tuple, Dict

import pandas as pd
import streamlit as st

# ---- Optional deps ----
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False


# ========= 共有ユーティリティ（本モジュール内で自給自足） =========
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell) -> List[str]:
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def split_multi(s) -> List[str]:
    """ '清酒; ワイン / ビール' などを分割 """
    if not s:
        return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    """列（文字列）に needles のいずれかが部分一致するか（小文字/全角空白正規化）。"""
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)


# ========= 共著エッジ作成 =========
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int, year_to: int,
                         targets: List[str] | None = None,
                         types: List[str] | None = None) -> pd.DataFrame:
    """
    入力: df（少なくとも '著者', '発行年', '対象物_top3', '研究タイプ_top3' を推奨）
    出力: edges DataFrame ['src', 'dst', 'weight']
    """
    use = df.copy()

    # 年で絞り込み
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # 対象物フィルタ
    if targets:
        if "対象物_top3" in use.columns:
            mask_tg = col_contains_any(use["対象物_top3"], targets)
            use = use[mask_tg]

    # 研究タイプフィルタ
    if types:
        if "研究タイプ_top3" in use.columns:
            mask_tp = col_contains_any(use["研究タイプ_top3"], types)
            use = use[mask_tp]

    # 著者ペアをカウント
    rows: List[Tuple[str, str]] = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = split_authors(a)
        for s, t in itertools.combinations(sorted(set(names)), 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]


# ========= 中心性スコア =========
def centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """
    edges: ['src','dst','weight']
    metric: 'degree'|'betweenness'|'eigenvector'
    返り値: ['author','score','coauth_count']
    """
    if edges.empty:
        return pd.DataFrame(columns=["author", "score", "coauth_count"])

    # 簡易共著数（重み和）だけは常に計算
    deg_simple = pd.concat([
        edges.groupby("src")["weight"].sum(),
        edges.groupby("dst")["weight"].sum(),
    ], axis=1).fillna(0)
    deg_simple["coauth_count"] = deg_simple["weight"].sum(axis=1)
    deg_simple = deg_simple["coauth_count"].reset_index().rename(columns={"index": "author"})

    if not HAS_NX:
        out = deg_simple.rename(columns={"coauth_count": "score"})
        return out[["author", "score", "coauth_count"]].sort_values("score", ascending=False).reset_index(drop=True)

    # networkx による中心性
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    cen_df = pd.Series(cen, name="score").reset_index().rename(columns={"index": "author"})
    out = pd.merge(cen_df, deg_simple, on="author", how="left")
    out["coauth_count"] = out["coauth_count"].fillna(0).astype(float)
    return out[["author", "score", "coauth_count"]].sort_values("score", ascending=False).reset_index(drop=True)


# ========= ウインドウスライス（時系列） =========
def _window_ranges(ymin: int, ymax: int, width: int, step: int) -> List[Tuple[int, int, int]]:
    """
    例: ymin=1990, ymax=2024, width=5, step=3
    -> [(1990,1994,1992), (1993,1997,1995), ...]  ※ (from,to,center)
    """
    out = []
    y = ymin
    while y <= ymax:
        y2 = min(y + width - 1, ymax)
        center = (y + y2) // 2
        out.append((y, y2, center))
        if y2 >= ymax:
            break
        y += step
    return out


def _timeseries_scores(df: pd.DataFrame,
                       ymin: int, ymax: int,
                       width: int, step: int,
                       metric: str,
                       targets: List[str], types: List[str],
                       top_n_each: int = 10,
                       max_authors: int = 20) -> pd.DataFrame:
    """
    期間をスライドしながら中心性スコアを算出。
    可視化用に ['center_year','author','score'] を返す。
    表示対象の author は各ウインドウの上位集合から最大 max_authors に制限。
    """
    windows = _window_ranges(ymin, ymax, width, step)
    records = []
    author_pool = []

    for yf, yt, yc in windows:
        edges = build_coauthor_edges(df, yf, yt, targets, types)
        rank = centrality_from_edges(edges, metric=metric)
        if rank.empty:
            continue
        # その窓の上位から候補を追加
        author_pool.extend(rank["author"].head(top_n_each).tolist())
        for _, r in rank.iterrows():
            records.append({"center_year": yc, "author": r["author"], "score": float(r["score"])})

    if not records:
        return pd.DataFrame(columns=["center_year", "author", "score"])

    ts = pd.DataFrame(records)
    # 可視化対象 author を制限（頻出上位）
    top_authors = ts["author"].value_counts().head(max_authors).index.tolist()
    ts = ts[ts["author"].isin(top_authors)].copy()
    ts = ts.sort_values(["author", "center_year"]).reset_index(drop=True)
    return ts


# ========= メイン描画 =========
def render_temporal_tab(df: pd.DataFrame) -> None:
    st.markdown("## ⏳ 研究ネットワークの経年変化")

    if df is None or "著者" not in df.columns:
        st.info("著者データが見つかりません。")
        return

    # 年の範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin_all, ymax_all = int(y.min()), int(y.max())
        else:
            ymin_all, ymax_all = 1980, 2025
    else:
        ymin_all, ymax_all = 1980, 2025

    # 1段目: 年・ウインドウ設定
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin_all, max_value=ymax_all,
                                 value=(ymin_all, ymax_all))
    with c2:
        win = st.number_input("ウインドウ幅（年）", min_value=2, max_value=15, value=5, step=1)
    with c3:
        step = st.number_input("ステップ（年）", min_value=1, max_value=10, value=2, step=1)
    with c4:
        metric = st.selectbox(
            "中心性指標",
            ["degree", "betweenness", "eigenvector"],
            index=0,
            format_func=lambda x: {
                "degree": "次数中心性（つながりの数）",
                "betweenness": "媒介中心性（橋渡し度）",
                "eigenvector": "固有ベクトル中心性（影響力）",
            }[x],
            help="networkx が未導入の場合は簡易スコア（共著数の合計）で代替します。",
        )

    # 2段目: 対象物・研究タイプフィルタ
    c5, c6 = st.columns([1, 1])
    with c5:
        # 候補抽出（表示順はアルファ順）
        tg_raw = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tg_all = sorted(tg_raw)
        tg_sel = st.multiselect("対象物で絞り込み（部分一致）", tg_all, default=[])
    with c6:
        tp_raw = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        tp_all = sorted(tp_raw)
        tp_sel = st.multiselect("研究タイプで絞り込み（部分一致）", tp_all, default=[])

    # 実行
    st.markdown("### 📈 中心性スコアの推移（スライドウインドウ）")
    ts = _timeseries_scores(
        df=df,
        ymin=y_from, ymax=y_to,
        width=int(win), step=int(step),
        metric=metric,
        targets=tg_sel, types=tp_sel,
        top_n_each=10, max_authors=20
    )

    if ts.empty:
        st.info("条件に合う共著ネットワークが構築できませんでした。年範囲やフィルタを調整してください。")
        return

    # 可視化
    if HAS_PX:
        fig = px.line(
            ts, x="center_year", y="score", color="author",
            markers=True,
            labels={"center_year": "年（ウインドウ中心）", "score": "中心性スコア", "author": "著者"},
        )
        fig.update_layout(legend_title_text="著者", height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(
            ts.pivot_table(index="center_year", columns="author", values="score", aggfunc="mean").sort_index()
        )

    # 直近ウインドウのランキング（参考）
    st.markdown("### 🔝 直近ウインドウの上位")
    last_from = max(y_to - int(win) + 1, y_from)
    last_to = y_to
    edges_last = build_coauthor_edges(df, last_from, last_to, tg_sel, tp_sel)
    rank_last = centrality_from_edges(edges_last, metric=metric).head(30)
    rank_last = rank_last.rename(columns={"author": "著者", "score": "中心性スコア", "coauth_count": "共著数"})
    st.dataframe(rank_last, use_container_width=True, hide_index=True)

    st.caption("※ 指標の意味：次数=つながりの数 / 媒介=橋渡し度 / 固有ベクトル=影響力（有力者との結び付き）")