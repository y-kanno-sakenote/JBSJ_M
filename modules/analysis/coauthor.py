# modules/analysis/coauthor.py
# -*- coding: utf-8 -*-
"""
共著ネットワーク（研究者のつながりランキング + ネットワーク可視化）
- 年・対象物・研究タイプでフィルタ（選択式）
- ランキング表：著者 / 共著数 / つながりスコア（中心性）
- 中心性指標は日本語表記で統一（次数中心性 / 媒介中心性 / 固有ベクトル中心性）
- ネットワーク描画は「ボタン」押下時のみ（PyVis / networkx があれば）
- PyVis 埋め込みは generate_html() を使用（ブラウザ自動起動を回避）
- サブタブ「⏳ 経年変化」は coauthor_temporal.py が存在する場合のみ自動で表示
"""

from __future__ import annotations
import re
import itertools
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ---- サブタブ（経年変化）の相対import：存在しない場合も落とさない ----
try:
    from .coauthor_temporal import render_coauthor_temporal_subtab  # 同ディレクトリ想定
    HAS_TEMPORAL = True
except Exception:
    HAS_TEMPORAL = False

# --- Optional deps ---
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

# --- 永続キャッシュIO（あれば使う・無くても動く） ---
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========= 基本ユーティリティ =========
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
_SPLIT_MULTI_RE  = re.compile(r"[;；,、，/／|｜\s　]+")

def split_authors(cell) -> List[str]:
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    """列（文字列）に needles のいずれかが部分一致するか（小文字・全角空白正規化）。"""
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)


# ========= 共著エッジ作成（フィルタ対応） =========
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int, year_to: int,
                         targets: List[str] | None = None,
                         types: List[str] | None = None) -> pd.DataFrame:
    """
    入力: df（少なくとも '著者', '発行年' を含むこと。対象物/研究タイプは任意）
    出力: edges DataFrame ['src', 'dst', 'weight']
    """
    use = df.copy()

    # 年で絞り込み
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # 対象物フィルタ（選択式）
    if targets:
        if "対象物_top3" in use.columns:
            mask_tg = col_contains_any(use["対象物_top3"], targets)
            use = use[mask_tg]

    # 研究タイプフィルタ（選択式）
    if types:
        if "研究タイプ_top3" in use.columns:
            mask_tp = col_contains_any(use["研究タイプ_top3"], types)
            use = use[mask_tp]

    # 著者のペアを数える
    rows: List[Tuple[str, str]] = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(split_authors(a)))
        for s, t in itertools.combinations(names, 2):
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
    返り値: ['著者','共著数','つながりスコア']
    """
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数", "つながりスコア"])

    # 共著数（重み和）は常に計算
    deg_simple = pd.concat([
        edges.groupby("src")["weight"].sum(),
        edges.groupby("dst")["weight"].sum(),
    ], axis=1).fillna(0)
    deg_simple["coauth_count"] = deg_simple["weight"].sum(axis=1)
    deg_simple = deg_simple["coauth_count"].reset_index().rename(columns={"index": "著者", "coauth_count": "共著数"})

    # networkx が無い場合は簡易スコア＝共著数
    if not HAS_NX:
        out = deg_simple.rename(columns={"共著数": "つながりスコア"})
        return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)

    # networkx による中心性
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r["weight"]))

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    cen_df = pd.Series(cen, name="つながりスコア").reset_index().rename(columns={"index": "著者"})
    out = pd.merge(cen_df, deg_simple, on="著者", how="left")
    out["共著数"] = out["共著数"].fillna(0).astype(float)
    return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)


# ========= ネットワーク描画（PyVis） =========
def _draw_network(edges: pd.DataFrame,
                  top_nodes: List[str] | None = None,
                  min_weight: int = 1,
                  height_px: int = 650) -> None:
    """PyVisで描画（任意）。依存が無ければスキップ。"""
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。表は利用できます。")
        return

    edges_use = edges[edges["weight"] >= int(min_weight)].copy()
    if edges_use.empty:
        st.warning("条件に合うエッジがありません。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if top_nodes:
        # グラフに存在するノードだけに限定
        top_nodes_in = [n for n in top_nodes if n in G]
        keep = set(top_nodes_in)
        # 近傍も含める（存在チェック付き）
        for n in top_nodes_in:
            for nbr in G.neighbors(n):
                keep.add(nbr)
        G = G.subgraph(keep).copy()
        if G.number_of_nodes() == 0:
            st.warning("トップNがグラフに存在しません。条件を見直してください。")
            return

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=110, spring_strength=0.02)
    net.from_nx(G)

    # ブラウザ自動オープンを避けて埋め込み
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)


# ========= コピー用の軽量HTMLグリッド =========
def _render_copy_grid(authors: List[str]) -> None:
    """表は崩さず、別枠で著者名のコピーUXを提供する小さなHTMLグリッド。"""
    if not authors:
        return
    html = """
    <style>
      .copy-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; }
      .copy-chip { display:flex; align-items:center; justify-content:space-between;
                   padding:6px 10px; background:#f5f5f7; border:1px solid #ddd; border-radius:8px; font-size:13px; }
      .copy-chip button { border:none; background:#e9e9ee; padding:4px 8px; border-radius:6px; cursor:pointer; }
      .copy-chip button:hover { background:#dcdce3; }
    </style>
    <div class="copy-grid">
    """
    for name in authors:
        safe_text = str(name).replace("\\", "\\\\").replace("'", "\\'")
        html += f"""
        <div class="copy-chip">
          <span>{safe_text}</span>
          <button onclick="navigator.clipboard.writeText('{safe_text}');
                           const n=document.createElement('div');
                           n.textContent='📋「{safe_text}」をコピーしました';
                           n.style='position:fixed;bottom:80px;right:30px;padding:10px 18px;background:#333;color:#fff;border-radius:8px;opacity:0.94;font-size:13px;z-index:9999';
                           document.body.appendChild(n); setTimeout(()=>n.remove(),1400);">
            📋
          </button>
        </div>
        """
    html += "</div>"
    import streamlit.components.v1 as components
    components.html(html, height=400, scrolling=True)


# ========= UI構築（サブタブ対応） =========
def render_coauthor_tab(df: pd.DataFrame, use_disk_cache: bool = False):
    st.markdown("## 👥 研究者のつながり分析（共著ネットワーク）")
    st.caption("共著関係が多いほどネットワークの中心に位置しやすく、橋渡し役や影響力の強さも指標から読み取れます。")

    if df is None or "著者" not in df.columns:
        st.warning("著者データが見つかりません。")
        return

    # タブ構成（経年変化サブタブはモジュールがあるときだけ）
    if HAS_TEMPORAL:
        tab_main, tab_temp = st.tabs(["🔝 ランキング・ネットワーク", "⏳ 経年変化"])
    else:
        (tab_main,) = st.tabs(["🔝 ランキング・ネットワーク"])

    # ===== メインタブ =====
    with tab_main:
        # 年範囲
        if "発行年" in df.columns:
            y = pd.to_numeric(df["発行年"], errors="coerce")
            if y.notna().any():
                ymin, ymax = int(y.min()), int(y.max())
            else:
                ymin, ymax = 1980, 2025
        else:
            ymin, ymax = 1980, 2025

        # フィルタ（選択式）
        targets_all = sorted({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)})
        types_all   = sorted({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)})

        c1, c2, c3= st.columns([1, 1, 1])
        with c1:
            year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
        with c2:
            tg_sel = st.multiselect("対象物で絞り込み", options=targets_all, default=[])
        with c3:
            tp_sel = st.multiselect("研究タイプで絞り込み", options=types_all, default=[])

        c4, c5, c6 = st.columns([1, 1, 1])
        with c4:
            metric = st.selectbox(
                "中心性指標",
                ["degree", "betweenness", "eigenvector"],
                index=0,
                format_func=lambda x: {
                    "degree": "次数（つながりの数）",
                    "betweenness": "媒介（橋渡し度）",
                    "eigenvector": "固有ベクトル（影響力）",
                }[x],
                help="networkx が未導入の場合は簡易スコア（共著数の合計）で代替します。",
            )
        with c5:
            top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5)
        with c6:
            min_w = st.number_input("描画する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1)

        # ---- キャッシュキー（オプション） ----
        cache_key = f"coauth_edges|{year_from}-{year_to}|tg{','.join(tg_sel)}|tp{','.join(tp_sel)}"
        edges = None
        if use_disk_cache and HAS_DISK_CACHE:
            path = cache_csv_path("coauthor_edges", cache_key)
            cached = load_csv_if_exists(path)
            if cached is not None:
                edges = cached

        if edges is None:
            edges = build_coauthor_edges(df, year_from, year_to, tg_sel, tp_sel)
            if use_disk_cache and HAS_DISK_CACHE:
                save_csv(edges, cache_csv_path("coauthor_edges", cache_key))

        if edges.empty:
            st.info("共著関係が見つかりませんでした。条件を調整してください。")
            return

        # --- スコア表示（表の仕様は維持） ---
        st.markdown("### 🔝 研究者のつながりランキング")
        rank = centrality_from_edges(edges, metric=metric).head(int(top_n))
        st.dataframe(rank, use_container_width=True, hide_index=True)
        st.caption("※ 指標の意味：次数=つながりの数 / 媒介=橋渡し度 / 固有ベクトル=影響力（有力者との結び付き）")

        # --- 補助：著者名のクイックコピー（別枠・表は崩さない） ---
        with st.expander("📋 著者名をすぐコピー（表はそのまま・補助機能）", expanded=False):
            _render_copy_grid(rank["著者"].tolist())

        # --- 可視化（遅延描画） ---
        with st.expander("🕸️ ネットワークを可視化（任意・依存あり）", expanded=False):
            st.caption("共著関係をインタラクティブに可視化します（networkx / pyvis が必要）。")
            top_only = st.toggle("上位ランキングの周辺だけ表示（軽量）", value=True)
            top_nodes = rank["著者"].tolist() if top_only else None
            if st.button("🌐 ネットワークを描画する"):
                _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)

    # ===== サブタブ：経年変化 =====
    if HAS_TEMPORAL:
        with tab_temp:
            render_coauthor_temporal_subtab(df, use_disk_cache=use_disk_cache)