# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
経年的変化：中心性の時系列（移動窓）可視化
- 年/対象物/研究タイプでフィルタ
- metric: degree / betweenness / eigenvector
- Plotlyが無い環境でも表は表示（任意依存）
- ディスクキャッシュ対応（.cache配下）
"""

from __future__ import annotations
from pathlib import Path
import re
import itertools
import pandas as pd
import streamlit as st

# Optional deps
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

# 共有
def split_authors(cell):
    if cell is None: return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜]+", str(cell)) if w.strip()]
def split_multi(s):
    if not s: return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

# ディスクキャッシュ
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
def _sig(*parts) -> str:
    import hashlib
    h = hashlib.md5()
    for p in parts: h.update(str(p).encode("utf-8"))
    return h.hexdigest()
def _cache_csv(prefix: str, *params) -> Path:
    return CACHE_DIR / f"{prefix}_{_sig(*params)}.csv"
def _load_csv(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try: return pd.read_csv(p)
        except Exception: return None
    return None
def _save_csv(df: pd.DataFrame, p: Path):
    try: df.to_csv(p, index=False)
    except Exception: pass

# ========== コア計算 ==========
def _apply_filters(df: pd.DataFrame,
                   year_from: int | None, year_to: int | None,
                   targets_sel: list[str] | None, types_sel: list[str] | None) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns and year_from is not None and year_to is not None:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]
    if targets_sel and "対象物_top3" in use.columns:
        keys = [k.lower() for k in targets_sel]
        use = use[use["対象物_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]
    if types_sel and "研究タイプ_top3" in use.columns:
        keys = [k.lower() for k in types_sel]
        use = use[use["研究タイプ_top3"].astype(str).str.lower().apply(lambda v: any(k in v for k in keys))]
    return use

def _edges_from_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for authors in df.get("著者", pd.Series(dtype=str)).fillna(""):
        names = split_authors(authors)
        u = sorted(set(names))
        for s, t in itertools.combinations(u, 2):
            rows.append((s, t))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src","dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    return edges.drop(columns=["pair"])

def _centrality_score(edges: pd.DataFrame, metric: str) -> pd.Series:
    """return Series(author -> score). networkx無い時は共著数合計で代替。"""
    if edges.empty:
        return pd.Series(dtype=float)

    # 代替（networkxなし）
    if not HAS_NX:
        deg = pd.concat([
            edges.groupby("src")["weight"].sum(),
            edges.groupby("dst")["weight"].sum(),
        ], axis=1).fillna(0)
        return deg.sum(axis=1).rename("score")

    # networkx あり
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)
    if len(G) == 0:
        return pd.Series(dtype=float)

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)
    return pd.Series(cen, name="score")

def _compute_centrality_over_windows(df: pd.DataFrame, metric: str, window: int) -> pd.DataFrame:
    """
    非重複の移動窓（window年）で中心性を算出。
    return: tidy DF ['year_start','year_end','author','score']
    """
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["year_start","year_end","author","score"])
    y = pd.to_numeric(df["発行年"], errors="coerce").dropna().astype(int)
    if y.empty:
        return pd.DataFrame(columns=["year_start","year_end","author","score"])
    ymin, ymax = int(y.min()), int(y.max())
    rows = []
    # 非重複（例: 2000-2004, 2005-2009 ...）
    for ys in range(ymin, ymax+1, window):
        ye = min(ys + window - 1, ymax)
        dwin = df[(pd.to_numeric(df["発行年"], errors="coerce").between(ys, ye))]
        edges = _edges_from_df(dwin)
        score = _centrality_score(edges, metric)
        if score.empty: 
            continue
        for author, val in score.items():
            rows.append((ys, ye, author, float(val)))
    out = pd.DataFrame(rows, columns=["year_start","year_end","author","score"])
    return out

def get_temporal_centrality(df: pd.DataFrame, metric: str, window: int,
                            year_from: int | None, year_to: int | None,
                            targets_sel: list[str] | None, types_sel: list[str] | None,
                            use_disk_cache: bool) -> pd.DataFrame:
    use = _apply_filters(df, year_from, year_to, targets_sel, types_sel)
    p = _cache_csv("temporal_centrality",
                   metric, window, len(use), year_from, year_to,
                   ",".join(sorted(targets_sel or [])),
                   ",".join(sorted(types_sel or [])))
    if use_disk_cache:
        cached = _load_csv(p)
        if cached is not None: return cached

    out = _compute_centrality_over_windows(use, metric=metric, window=window)
    if use_disk_cache and not out.empty:
        _save_csv(out, p)
    return out

# ========== UI ==========
def render_temporal_tab(df: pd.DataFrame, use_disk_cache: bool = False):
    st.markdown("## ⏳ 研究者の“中心度”の経年変化")

    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin, ymax = int(y.min()), int(y.max())
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    kpref = "tmp_"
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                       value=(ymin, ymax), key=f"{kpref}yr")
    with c2:
        metric = st.selectbox("中心性の種類", ["degree", "betweenness", "eigenvector"], index=0, key=f"{kpref}met",
                              help="networkx未導入時は共著数合計を代替スコアとして利用")
    with c3:
        window = st.selectbox("窓幅（年）", [3,5,10], index=1, key=f"{kpref}win")
    with c4:
        top_k = st.number_input("可視化する上位人数", min_value=3, max_value=50, value=10, step=1, key=f"{kpref}k")

    c5, c6 = st.columns([1,1])
    with c5:
        targets_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets_sel = st.multiselect("対象物（部分一致）", targets_all, default=[], key=f"{kpref}tg")
    with c6:
        types_all = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types_sel = st.multiselect("研究タイプ（部分一致）", types_all, default=[], key=f"{kpref}tp")

    # 計算（キャッシュあり）
    tidy = get_temporal_centrality(df, metric=metric, window=int(window),
                                   year_from=year_from, year_to=year_to,
                                   targets_sel=targets_sel, types_sel=types_sel,
                                   use_disk_cache=use_disk_cache)
    if tidy.empty:
        st.info("対象期間・条件で有効な共著データがありません。条件を調整してください。")
        return

    # 期間の代表年（中心）をx軸用に
    tidy["year_mid"] = (tidy["year_start"] + tidy["year_end"]) / 2

    # 上位著者選定（全期間平均スコア）
    top_authors = (tidy.groupby("author")["score"].mean()
                   .sort_values(ascending=False).head(int(top_k)).index.tolist())
    sub = tidy[tidy["author"].isin(top_authors)].copy()

    st.markdown("### 📈 中心性の推移（上位）")
    if HAS_PX:
        fig = px.line(sub, x="year_mid", y="score", color="author",
                      markers=True, labels={"year_mid":"年","score":"中心性スコア","author":"著者"},
                      title="研究者の中心性推移（移動窓）")
        fig.update_layout(legend_title_text="著者", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(sub.pivot_table(index="year_mid", columns="author", values="score")
                     .sort_index(), use_container_width=True)

    with st.expander("結果データ（tidy）", expanded=False):
        st.dataframe(sub[["year_start","year_end","author","score"]]
                     .sort_values(["author","year_start"]), use_container_width=True, hide_index=True)