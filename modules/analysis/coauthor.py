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


# ========= 年レンジユーティリティ =========
@st.cache_data(ttl=600, show_spinner=False)
def _year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

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


# ========= 並び順（temporal.py と統一） & 補助ソート関数 =========
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス",
    "酵母・微生物","アミノ酸・タンパク質","その他"
]

TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

def _sort_with_order(items: List[str], order: List[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(order)}
    # 未定義項目は末尾・元の名前順
    return sorted(items, key=lambda x: (order_map.get(x, len(order)), x))


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


# -------- 共通フィルタバー（外部 or 内蔵フォールバック） --------
try:
    # 他モジュールの共通フィルタ（存在すれば利用）
    from modules.common.filters import render_filter_bar  # type: ignore
except Exception:
    # フォールバック：このファイル内で簡易版を提供（UIは最小限）
    def render_filter_bar(df: pd.DataFrame, key_prefix: str = "authors",
                          show_presets: bool = False, sticky: bool = False):
        """共通フィルターバーの簡易版（年・対象物・研究タイプ）。"""
        ymin, ymax = _year_min_max(df)

        # 候補抽出（表示順は指定配列を優先）
        tg_all = sorted({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                         for w in split_multi(v) if w.strip()})
        tp_all = sorted({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                         for w in split_multi(v) if w.strip()})
        tg_all = _sort_with_order(list(tg_all), TARGET_ORDER)
        tp_all = _sort_with_order(list(tp_all), TYPE_ORDER)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            y_from, y_to = st.slider(
                "対象年（範囲）",
                min_value=ymin, max_value=ymax,
                value=(ymin, ymax),
                key=f"{key_prefix}_year",
            )
        with c2:
            tg_sel = st.multiselect(
                "対象物で絞り込み（部分一致）",
                options=tg_all, default=[],
                key=f"{key_prefix}_tg",
            )
        with c3:
            tp_sel = st.multiselect(
                "研究タイプで絞り込み（部分一致）",
                options=tp_all, default=[],
                key=f"{key_prefix}_tp",
            )

        return {"year": (y_from, y_to), "targets": tg_sel, "types": tp_sel}

# ======== フィルタバー結果のアダプタ ========
def _adapt_filter_bar(df: pd.DataFrame):
    """
    共通filters.render_filter_barの戻り値の差異を吸収するアダプタ。
    - 期待: dict {"year":(from,to), "targets":[...], "types":[...]}
    - 互換: DataFrame（既にフィルタ済み）を返す実装にも対応
    返り値: (df_use, y_from, y_to, targets, types)
    """
    # まず安全に呼び出す（外部filtersが予期しない引数を受け取らない場合に対応）
    try:
        res = render_filter_bar(df, key_prefix="authors", show_presets=True, sticky=True)
    except TypeError:
        try:
            res = render_filter_bar(df, key_prefix="authors")
        except Exception:
            res = df

    # dict 形式ならそのまま取り出し
    if isinstance(res, dict):
        y_from, y_to = res.get("year", _year_min_max(df))
        tg_sel = res.get("targets", [])
        tp_sel = res.get("types", [])
        df_use = _apply_filters_basic(df, y_from, y_to, tg_sel, tp_sel)
        return df_use, int(y_from), int(y_to), list(tg_sel), list(tp_sel)

    # DataFrame 形式ならフィルタ済みとして扱う
    if isinstance(res, pd.DataFrame):
        df_use = res
        # 年範囲はフィルタ後のdfから推定
        if "発行年" in df_use.columns:
            y = pd.to_numeric(df_use["発行年"], errors="coerce")
            if y.notna().any():
                y_from, y_to = int(y.min()), int(y.max())
            else:
                y_from, y_to = _year_min_max(df)
        else:
            y_from, y_to = _year_min_max(df)
        return df_use, y_from, y_to, [], []

    # それ以外は元dfを返す（フォールバック）
    y_from, y_to = _year_min_max(df)
    return df, y_from, y_to, [], []


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


# ========= 著者カウント系ユーティリティ =========
@st.cache_data(ttl=600, show_spinner=False)
def _apply_filters_basic(df: pd.DataFrame, y_from: int, y_to: int,
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

@st.cache_data(ttl=600, show_spinner=False)
def _author_total_counts(df: pd.DataFrame) -> pd.Series:
    """著者別の論文件数（重複同一論文内は1カウント）"""
    if "著者" not in df.columns:
        return pd.Series(dtype=int)
    bags = []
    for a in df["著者"].fillna(""):
        names = list(dict.fromkeys(split_authors(a)))
        bags += names
    if not bags:
        return pd.Series(dtype=int)
    s = pd.Series(bags, dtype="object")
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def _yearly_author_counts(df: pd.DataFrame) -> pd.DataFrame:
    """年×著者の件数（同一論文内の重複は1としてカウント）"""
    if "著者" not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "著者", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        names = list(dict.fromkeys(split_authors(r.get("著者", ""))))
        for n in names:
            rows.append((int(y), n))
    if not rows:
        return pd.DataFrame(columns=["発行年", "著者", "count"])
    c = pd.DataFrame(rows, columns=["発行年","著者"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)


# ========= ネットワーク描画（PyVis） =========
def _draw_network(edges: pd.DataFrame,
                  top_nodes: List[str] | None = None,
                  min_weight: int = 1,
                  height_px: int = 650) -> None:
    """PyVisで描画。依存が無ければスキップ。"""
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。表は利用できます。")
        return

    # --- 重みしきい値の適用 ---
    edges_use = edges[edges["weight"] >= int(min_weight)].copy()
    if edges_use.empty:
        st.warning("条件に合うエッジがありません。")
        return

    # --- NetworkX グラフ構築（重み付き） ---
    G = nx.Graph()
    for _, r in edges_use.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    # 上位ノードの周辺だけを表示（オプション）
    if top_nodes:
        top_nodes_in = [n for n in top_nodes if n in G]
        keep = set(top_nodes_in)
        for n in top_nodes_in:
            for nbr in G.neighbors(n):
                keep.add(nbr)
        G = G.subgraph(keep).copy()
        if G.number_of_nodes() == 0:
            st.warning("トップNがグラフに存在しません。条件を見直してください。")
            return

    # --- ノード強さ（接続重みの総和）と上位ラベル制御 ---
    import math
    strength = {}
    for n in G.nodes():
        wsum = 0.0
        for _, _, d in G.edges(n, data=True):
            wsum += float(d.get("weight", 1.0))
        strength[n] = wsum

    # ラベルは強いノード上位だけ表示（混雑回避）
    label_top = set(sorted(G.nodes(), key=lambda x: strength.get(x, 0.0), reverse=True)[:40])

    # --- コミュニティ（色分け）：失敗時は単色 ---
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
        comm_id = {}
        for i, cset in enumerate(comms):
            for n in cset:
                comm_id[n] = i
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}

    # カラーパレット（循環）
    palette = [
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
        "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#8c6d31"
    ]

    # --- PyVisへ流し込み（カスタム属性を反映） ---
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    # 物理・描画オプション（JSON 文字列として渡す：無効な JSON を避ける）
    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.25,
              "springLength": 110,
              "springConstant": 0.02,
              "damping": 0.30
            },
            "minVelocity": 0.75,
            "solver": "barnesHut",
            "stabilization": {
              "enabled": true,
              "fit": true,
              "iterations": 800
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "zoomView": true,
            "dragView": true
          },
          "nodes": {
            "shape": "dot",
            "borderWidth": 1
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "roundness": 0.2
            }
          }
        }
        """
    )

    # ノード追加（サイズ=log1p(強さ)、タイトルに強さ）
    for n in G.nodes():
        wsum = strength.get(n, 0.0)
        size = 8.0 + 4.0 * math.log1p(wsum)  # 線形だと極端になりやすいので log
        title = f"{n}｜総共著重み: {int(wsum)}"
        color = palette[comm_id.get(n, 0) % len(palette)]
        label = n if n in label_top else ""  # 上位だけラベル
        net.add_node(n, label=label, title=title, size=size, color=color)

    # エッジ追加（太さ=log1p(weight)、ホバーで回数表示）
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        width = 1.0 + math.log1p(w)
        title = f"共著回数: {int(w)}"
        net.add_edge(u, v, value=w, width=width, title=title)

    # 埋め込み（ブラウザ自動オープン回避）
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)
    # ダウンロード（単独HTMLとして保存）
    st.download_button(
        "📥 ネットワークHTML",
        data=html.encode("utf-8"),
        file_name="coauthor_network.html",
        mime="text/html",
        key="dl_coauthor_html",
        help="このネットワークを単独のHTMLファイルとして保存します（ブラウザでそのまま開けます）。"
    )


# ========= コピー用の軽量HTMLグリッド =========
def _render_copy_grid(authors: List[str]) -> None:
    """表は崩さず、別枠で著者名のコピーUXを提供する小さなHTMLグリッド。"""
    if not authors:
        return
    html = """
    <style>
      .copy-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px; }
      .copy-chip { display:flex; align-items:center; justify-content:space-between;
                   padding:4px 8px; background:#f5f5f7; border:1px solid #ddd; border-radius:8px; font-size:12px; }
      .copy-chip button { border:none; background:#e9e9ee; padding:3px 6px; border-radius:6px; cursor:pointer; }
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
    components.html(html, height=140, scrolling=True)


# ========= UI構築 =========
def render_coauthor_tab(df: pd.DataFrame, use_disk_cache: bool = False):
    # ===== タブ見出し（下揃え＋横並び） =====
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin: 0 0 4px 0;">
          <h2 style="margin:0; line-height:1; font-weight:600;">👨‍🔬 研究者</h2>
          <div style="margin:0 0 2px 0; line-height:1.2; opacity:0.8; font-size:0.95rem;">
            著者別の論文数・共著ネットワーク・トレンドを確認できます。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None or ("著者" not in df.columns):
        st.warning("著者データが見つかりません。")
        return

    # 共通フィルターバー（年・対象物・研究タイプ）: アダプタで取得
    df_use, y_from, y_to, tg_sel, tp_sel = _adapt_filter_bar(df)

    # サブタブ構成：①論文数 ②共著ネットワーク ③トレンド分析
    tab_count, tab_network, tab_trend = st.tabs(["① 論文数", "② 共著ネットワーク", "③ トレンド分析"])

    # ===== ① 論文数 =====
    with tab_count:
        # --- 詳細フィルタ行（ランキング件数 + 集計期間 + 単著/共著 + 著者ポジション） ---
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            mode = st.radio(
                "著者数フィルタ",
                ["すべて", "単著のみ", "共著のみ"],
                horizontal=True,
                key="res_cnt_mode",
                label_visibility="visible",
            )

        with c2:
            period = st.radio(
                "集計期間",
                ["累計", "直近1年", "直近3年", "直近5年"],
                horizontal=True,
                key="res_cnt_period",
                label_visibility="visible",
            )
        with c3:
            position = st.multiselect("著者ポジション", ["筆頭のみ","責任著者のみ"], key="res_cnt_position")
        with c4:
            top_n = st.number_input("ランキング件数", min_value=5, max_value=200, value=50, step=5, key="res_cnt_topn")

        # --- 期間・著者数・ポジションでフィルタリング ---
        df_rank = df_use

        # 期間フィルタ
        if period != "累計" and "発行年" in df_rank.columns:
            years = pd.to_numeric(df_rank["発行年"], errors="coerce")
            span = {"直近1年":1, "直近3年":3, "直近5年":5}[period]
            y_max = int(years.max()) if years.notna().any() else None
            if y_max is not None:
                df_rank = df_rank[(years >= y_max - span + 1) & (years <= y_max)]

        # 単著/共著フィルタ
        if mode != "すべて":
            df_rank = df_rank.copy()
            df_rank["著者数"] = df_rank["著者"].fillna("").map(lambda s: len(split_authors(s)))
            if mode == "単著のみ":
                df_rank = df_rank[df_rank["著者数"] == 1]
            else:
                df_rank = df_rank[df_rank["著者数"] >= 2]

        # 筆頭/責任著者フィルタは、カウント時に安全に処理するため、ここでは除外

        # --- データ準備（フィルタ適用後の著者ランキング） ---
        use = df_rank
        # 位置指定がある場合は、カウント段階で筆頭/責任著者のみを加算
        if position:
            bags = []
            for _, r in df_rank.iterrows():
                names = list(dict.fromkeys(split_authors(r.get("著者", ""))))
                if not names:
                    continue
                chosen = []
                if "筆頭のみ" in position and len(names) >= 1:
                    chosen.append(names[0])
                if "責任著者のみ" in position and len(names) >= 1:
                    chosen.append(names[-1])
                # 両方同じ場合（単著など）は重複排除
                if chosen:
                    bags.extend(list(dict.fromkeys(chosen)))
            if bags:
                s = pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)
            else:
                s = pd.Series(dtype=int)
        else:
            s = _author_total_counts(df_rank)
        if s.empty:
            st.info("条件に合うデータがありません。")
        else:
            rank = s.reset_index()
            rank.columns = ["著者", "論文数"]


            # 並び順は常に 論文数降順 → 著者名（同数時）
            rank = rank.sort_values(["論文数", "著者"], ascending=[False, True])

            # 表示件数の制御（右の表に適用）。バーは上位20を左に表示。
            rank_shown = rank.head(int(top_n))

            # ① 左右2ペイン（左：表 / 右：棒グラフ）
            left, right = st.columns([1.0, 1.1])
            with left:
                st.dataframe(
                    rank_shown[["著者", "論文数"]],
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

            with right:
                try:
                    import plotly.express as px
                    # 横棒：上位10を降順で。上から大きい順に並ぶようにする
                    bar_df = rank.head(10).sort_values("論文数", ascending=False)
                    fig = px.bar(
                        bar_df,
                        x="論文数",
                        y="著者",
                        orientation="h",
                        text_auto=True,
                        title="著者Top10",
                    )
                    fig.update_layout(
                        margin=dict(l=6, r=6, t=40, b=6),
                        height=420,
                        xaxis_title=None,
                        yaxis_title=None,
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    # フォールバック：水平にできない場合でも上位10を表示
                    st.bar_chart(rank.set_index("著者")["論文数"].head(10))

            # ④ クイックコピー：現在表示行（rank_shown）の著者だけ
            with st.expander("📋 著者名をすぐコピー", expanded=False):
                _render_copy_grid(rank_shown["著者"].tolist())

            # ⑥ 対象物別のTop5著者（改善版UI）
            with st.expander("🏷️ 対象物別のTop5著者（現在のフィルタで集計）", expanded=False):
                # ▼ 見やすさ改善版：対象物ごとのTop5を「横棒グラフの小カード」で並べる（最大8グループ）
                view_mode = st.radio("表示形式", ["グラフ", "表"], horizontal=True, key="res_cnt_tg_view")
                try:
                    # 対象物ごとに著者カウント
                    rows = []
                    for _, r in use.iterrows():
                        tg_list = list(dict.fromkeys(split_multi(r.get("対象物_top3", ""))))
                        names = list(dict.fromkeys(split_authors(r.get("著者", ""))))
                        for tg in tg_list:
                            for n in names:
                                if tg and n:
                                    rows.append((tg, n))
                    if not rows:
                        st.caption("対象物別の上位情報はありません。")
                    else:
                        df_tg = pd.DataFrame(rows, columns=["対象物", "著者"]).value_counts().reset_index(name="件数")
                        # 多すぎる対象物は上位のものだけ表示（最大8グループ）
                        heads = df_tg.groupby("対象物")["件数"].sum().sort_values(ascending=False).head(8).index.tolist()
                        show = (
                            df_tg[df_tg["対象物"].isin(heads)]
                            .sort_values(["対象物", "件数"], ascending=[True, False])
                            .groupby("対象物")
                            .head(5)
                            .reset_index(drop=True)
                        )

                        # ダウンロード（CSV）
                        st.download_button(
                            "📥 この一覧をCSVで保存",
                            data=show.to_csv(index=False).encode("utf-8"),
                            file_name="target_top5_authors.csv",
                            mime="text/csv",
                            key="dl_target_top5_authors"
                        )

                        if view_mode == "表":
                            st.dataframe(show, use_container_width=True, hide_index=True)
                        else:
                            try:
                                import plotly.express as px
                                # 対象物ごとに2列のカード配置で可読性UP
                                cols = st.columns(2)
                                for i, tg in enumerate(heads):
                                    sub = show[show["対象物"] == tg].copy()
                                    # 横棒用に並べ替え（小さい→大きいで積み上がる視覚を作る）
                                    sub = sub.sort_values("件数", ascending=True)
                                    with cols[i % 2]:
                                        fig = px.bar(
                                            sub,
                                            x="件数",
                                            y="著者",
                                            orientation="h",
                                            text_auto=True,
                                            title=tg
                                        )
                                        fig.update_layout(
                                            height=260,
                                            margin=dict(l=8, r=8, t=36, b=8),
                                            xaxis_title=None,
                                            yaxis_title=None
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                # Plotlyが無い場合は対象物ごとに小さな表で代替
                                cols = st.columns(2)
                                for i, tg in enumerate(heads):
                                    sub = show[show["対象物"] == tg].sort_values("件数", ascending=False)
                                    with cols[i % 2]:
                                        st.markdown(f"**{tg}**")
                                        st.dataframe(sub[["著者", "件数"]], use_container_width=True, hide_index=True)
                except Exception as e:
                    st.caption(f"対象物別Topの集計に失敗しました: {e!s}")

    # ===== ② 共著ネットワーク（既存ロジックをそのまま） =====
    with tab_network:
        # メトリック・ランキング件数・最小共著回数 のみ
        c4, c5, c6 = st.columns([1,1,1])
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
                key="res_net_metric",
            )
        with c5:
            top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5, key="res_net_topn")
        with c6:
            min_w = st.number_input("描画する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1, key="res_net_minw")

        # エッジ構築（ディスクキャッシュは従来どおり利用可）
        _tg_key = ",".join(tg_sel) if tg_sel else ""
        _tp_key = ",".join(tp_sel) if tp_sel else ""
        cache_key = f"coauth_edges|{y_from}-{y_to}|tg{_tg_key}|tp{_tp_key}"
        edges = None
        if use_disk_cache and HAS_DISK_CACHE:
            path = cache_csv_path("coauthor_edges", cache_key)
            cached = load_csv_if_exists(path)
            if cached is not None:
                edges = cached
        if edges is None:
            edges = build_coauthor_edges(df_use, y_from, y_to, tg_sel, tp_sel)
            if use_disk_cache and HAS_DISK_CACHE:
                save_csv(edges, cache_csv_path("coauthor_edges", cache_key))

        if edges.empty:
            st.info("共著関係が見つかりませんでした。条件を調整してください。")
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:6px; margin:6px 0 2px 0;">
                  <span style="font-weight:600; font-size:0.95rem; opacity:0.9;">🔝 研究者のつながりランキング</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            rank = centrality_from_edges(edges, metric=metric).head(int(top_n))
            st.dataframe(rank, use_container_width=True, hide_index=True)
            st.caption("※ 指標の意味：次数=つながりの数 / 媒介=橋渡し度 / 固有ベクトル=影響力（有力者との結び付き）")
            with st.expander("📋 著者名をすぐコピー", expanded=False):
                _render_copy_grid(rank["著者"].tolist())

            with st.expander("🕸️ ネットワークを可視化", expanded=False):
                top_only = st.toggle("上位ランキングの周辺だけ表示", value=True, key="res_net_toponly")
                top_nodes = rank["著者"].tolist() if top_only else None
                if st.button("🌐 描画する", key="res_net_draw"):
                    _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)

    # ===== ③ トレンド分析（論文数の年次推移） =====
    with tab_trend:
        use = df_use
        yearly = _yearly_author_counts(use)
        if yearly.empty:
            st.info("データがありません。")
            return

        # 著者候補（出現総数の多い順）
        tot = yearly.groupby("著者")["count"].sum().sort_values(ascending=False)
        options = tot.index.tolist()

        # --- 詳細フィルターを1列で並べる：初期表示の著者数 → 表示する著者 → 移動平均 ---
        col_a, col_b, col_c = st.columns([1, 7, 1])

        with col_a:
            max_auth = st.number_input(
                "初期表示数（上位）",
                min_value=3, max_value=30, value=10, step=1,
                key="res_trend_initn"
            )

        default_sel = options[: int(max_auth)]

        with col_b:
            sel = st.multiselect(
                "表示する著者（複数可）",
                options,
                default=default_sel,
                key="res_trend_authors"
            )

        with col_c:
            ma = st.number_input(
                "移動平均（年）",
                min_value=1, max_value=7, value=1, step=1,
                key="res_trend_ma"
            )

        piv = yearly.pivot_table(index="発行年", columns="著者", values="count", aggfunc="sum").fillna(0).sort_index()
        if sel:
            piv = piv[[c for c in sel if c in piv.columns]]

        if piv.shape[1] == 0:
            st.info("表示対象がありません。左のリストから1つ以上選んでください。")
            return

        if int(ma) > 1:
            piv = piv.rolling(window=int(ma), min_periods=1).mean()

        # --- ▼▼▼ ここから新規UI（表示指標/凡例順序）挿入 ▼▼▼ ---
        # 表示指標: 件数 / シェア(%)
        metric_mode = st.radio(
            "表示指標", ["件数", "シェア(%)"], horizontal=True, key="res_trend_metric",
            help="シェア(%)を選ぶと、各年内での著者の占有率を表示します。"
        )


        # シェア化
        if metric_mode == "シェア(%)":
            row_sums = piv.sum(axis=1)
            piv = piv.div(row_sums, axis=0).fillna(0) * 100

        # 凡例順序を直近年値で並べ替え（デフォルトで常時適用）
        if not piv.empty:
            try:
                last_row = piv.iloc[-1]  # 最終行（直近年）
            except Exception:
                last_row = piv.mean(axis=0, numeric_only=True)
            order = list(last_row.sort_values(ascending=False).index)
            piv = piv.loc[:, [c for c in order if c in piv.columns]]
        # --- ▲▲▲ ここまで挿入 --- 

        # ⭐ 直近上昇ハイライト（トグルはグラフの“下”に移動）
        hi_key = "res_trend_hi"
        hi_on = bool(st.session_state.get(hi_key, False))

        # 凡例名マップ（デフォルトは同名）
        legend_map = {c: c for c in piv.columns}
        highlighted = []

        if hi_on and not piv.empty:
            years = list(piv.index)
            # 直近3年とその直前3年（不足時はある分で計算）
            recent = [y for y in years if y <= y_to][-3:]
            prev   = [y for y in years if y < (recent[0] if recent else y_to)][-3:]

            def growth_for(col: str) -> float:
                r = float(piv.loc[recent, col].mean()) if recent else 0.0
                p = float(piv.loc[prev,   col].mean()) if prev else 0.0
                return (r + 1.0) / (p + 1.0) - 1.0

            scores = {c: growth_for(c) for c in piv.columns}
            # 上位5名に⭐付与
            top_names = [k for k, _ in sorted(scores.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -1e9), reverse=True)[:5]]
            for n in top_names:
                legend_map[n] = f"⭐ {n}"
            highlighted = top_names

        # グラフの“下”で表示するための例文を先に用意
        highlight_example_text = ""
        if hi_on and highlighted:
            highlight_example_text = "⭐ 直近上昇ハイライト: " + ", ".join(legend_map[n] for n in highlighted)

        try:
            import plotly.express as px
            _sel_key = ",".join(sel) if sel else "__ALL__"
            _uniq_key = f"res_trend_plot|{y_from}-{y_to}|{_sel_key}|ma{ma}|hi{int(hi_on)}|m{metric_mode}"
            plot_df = piv.reset_index().melt(id_vars="発行年", var_name="著者", value_name="値")
            # 凡例表示名を差し替え（⭐ 付与）
            plot_df["著者"] = plot_df["著者"].map(legend_map).fillna(plot_df["著者"])
            y_axis_title = metric_mode if metric_mode == "件数" else "シェア(%)"
            fig = px.line(plot_df, x="発行年", y="値", color="著者", markers=True)
            fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), legend_title_text="著者", yaxis_title=y_axis_title)
            st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
        except Exception:
            # フォールバック（凡例名の差し替えは不可だがグラフは表示）
            st.line_chart(piv)
        # グラフの“下”にトグルと例を横並び表示（標準ウィジェットのみ・デフォルト形）
        col_tgl, col_example = st.columns([0.22, 0.78])
        with col_tgl:
            hi_on_new = st.toggle(
                "⭐ 直近上昇をハイライト",
                value=hi_on,
                key=hi_key,
                help="直近3年平均とその前3年平均を比較して、増加の大きい著者に⭐を付けます。"
            )
            # ※ 値は Streamlit が管理。代入は不要。
        with col_example:
            # 少し下にずらすための薄いスペーサ
            st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
            if hi_on_new and highlight_example_text:
                st.caption(highlight_example_text)

        # ▼ 著者名コピー（補助機能）：表示中の著者だけを並べる
        with st.expander("📋 著者名をすぐコピー", expanded=False):
            try:
                current_authors = list(piv.columns)
            except Exception:
                current_authors = sel if isinstance(sel, list) else []
            _render_copy_grid(current_authors)