# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化）
- 年/対象物/研究タイプでフィルタ
- 表: 著者 / 共著数 / つながりスコア（中心性） の3列
- ネットワーク描画は PyVis（generate_html で安定埋め込み）
- ディスクキャッシュ対応（.cache配下）
"""

from __future__ import annotations
import re
import itertools
from pathlib import Path
import pandas as pd
import streamlit as st

# --- Optional deps ---
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

# 共有ユーティリティ（簡易）
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
def split_authors(cell):
    if cell is None: return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(s):
    if not s: return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

# ---- ディスクキャッシュ（共通ユーティリティ最小版） ----
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
def _sig(*parts) -> str:
    import hashlib
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode("utf-8"))
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

# ========= エッジ生成 =========
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame) -> pd.DataFrame:
    """df（著者列はフィルタ済みのもの）→ edges[src,dst,weight]"""
    if df is None or "著者" not in df.columns:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    rows = []
    for authors in df["著者"].fillna(""):
        names = split_authors(authors)
        uniq = sorted(set(names))
        for s, t in itertools.combinations(uniq, 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    # 無向グラフなので pair をソートして集計
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]

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

def get_coauthor_edges(df: pd.DataFrame,
                       year_from: int | None, year_to: int | None,
                       targets_sel: list[str] | None,
                       types_sel: list[str] | None,
                       use_disk_cache: bool) -> pd.DataFrame:
    """フィルタ込みでエッジ作成＋CSVキャッシュ"""
    use = _apply_filters(df, year_from, year_to, targets_sel, types_sel)
    keypath = _cache_csv("coauthor_edges",
                         len(use), year_from, year_to,
                         ",".join(sorted(targets_sel or [])),
                         ",".join(sorted(types_sel or [])))
    if use_disk_cache:
        cached = _load_csv(keypath)
        if cached is not None: return cached

    edges = build_coauthor_edges(use)
    if use_disk_cache and not edges.empty:
        _save_csv(edges, keypath)
    return edges

# ========= 中心性 =========
def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    """
    input: edges[src,dst,weight]
    output: DataFrame['著者','共著数','つながりスコア']
    - 共著数: 接続エッジ重みの合計（簡易な“関与度”）
    - つながりスコア: degree/betweenness/eigenvector (networkx無い場合は共著数をそのまま)
    """
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数", "つながりスコア"])

    # 共著数（重み合計）
    deg = pd.concat([
        edges.groupby("src")["weight"].sum(),
        edges.groupby("dst")["weight"].sum(),
    ], axis=1).fillna(0)
    deg["共著数"] = deg["weight"].sum(axis=1)
    deg = deg[["共著数"]]

    if not HAS_NX:
        out = deg.sort_values("共著数", ascending=False).reset_index().rename(columns={"index": "著者"})
        out["つながりスコア"] = out["共著数"]  # 代替
        return out

    # networkx で中心性
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    cen = pd.Series(cen, name="つながりスコア")
    out = deg.join(cen, how="outer").fillna(0).reset_index().rename(columns={"index": "著者"})
    out = out.sort_values(["つながりスコア", "共著数"], ascending=False).reset_index(drop=True)
    return out

# ========= 可視化 =========
def _draw_network(edges: pd.DataFrame, top_nodes=None, min_weight=1, height_px=650):
    """PyVisで安定埋め込み（generate_html使用）"""
    if not (HAS_NX and HAS_PYVIS):
        st.info("⚠️ グラフ描画には networkx / pyvis が必要です。ランキング表は利用できます。")
        return
    edges_use = edges[edges["weight"] >= int(min_weight)]
    if edges_use.empty:
        st.warning("条件に合う共著関係が見つかりませんでした。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        # “存在しないノード”を弾く
        top_nodes = [n for n in (top_nodes or []) if n in G]
        keep = set(top_nodes) | {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()
        if len(G) == 0:
            st.warning("トップ近傍にエッジがありません。閾値を下げてください。")
            return

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#fff", font_color="#222")
    net.barnes_hut(gravity=-25000, central_gravity=0.25, spring_length=110, spring_strength=0.02)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    net.set_options('{"nodes":{"shape":"dot","scaling":{"min":10,"max":40}},"edges":{"smooth":false}}')

    # ★ ブラウザ自動オープンを回避してHTML文字列を直接埋め込む
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)

# ========= UI =========
def render_coauthor_tab(df: pd.DataFrame, use_disk_cache: bool = False):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク）")

    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            ymin, ymax = int(y.min()), int(y.max())
        else:
            ymin, ymax = 1980, 2025
    else:
        ymin, ymax = 1980, 2025

    # フィルタUI（キー衝突回避のため接頭辞）
    kpref = "co_"
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key=f"{kpref}yr")
    with c2:
        metric = st.selectbox("スコア計算方式", ["degree", "betweenness", "eigenvector"], index=0, key=f"{kpref}met")
    with c3:
        top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5, key=f"{kpref}n")
    with c4:
        min_w = st.number_input("共著回数の下限 (w≥)", min_value=1, max_value=20, value=2, step=1, key=f"{kpref}mw")

    # 対象物/研究タイプ（候補抽出はシンプルに）
    c5, c6 = st.columns([1, 1])
    with c5:
        targets_all = sorted({t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        targets_sel = st.multiselect("対象物（部分一致）", targets_all, default=[], key=f"{kpref}tg")
    with c6:
        types_all = sorted({t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        types_sel = st.multiselect("研究タイプ（部分一致）", types_all, default=[], key=f"{kpref}tp")

    # エッジ作成（キャッシュ考慮）
    edges = get_coauthor_edges(df, y_from, y_to, targets_sel, types_sel, use_disk_cache=use_disk_cache)
    if edges.empty:
        st.info("共著関係が見つかりませんでした。条件を緩めてお試しください。")
        return

    # ランキング（著者 / 共著数 / つながりスコア）
    st.markdown("### 🔝 研究者のつながりランキング")
    rank = _centrality_from_edges(edges, metric=metric).head(int(top_n))
    st.dataframe(rank[["著者", "共著数", "つながりスコア"]], use_container_width=True, hide_index=True)

    # 可視化
    with st.expander("🕸️ ネットワークを可視化（任意）", expanded=False):
        st.caption("PyVisでインタラクティブに表示します（networkx/pyvis が必要）")
        top_only = st.toggle("トップNの周辺のみ表示（軽量）", value=True, key=f"{kpref}toponly")
        top_nodes = rank["著者"].tolist() if top_only else None
        if st.button("🌐 描画する", key=f"{kpref}draw"):
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)