# modules/analysis_tab.py
# -*- coding: utf-8 -*-
"""
分析タブ（共著ネットワーク：中心性ランキング + ネットワーク可視化）
- 依存ゼロで「中心性ランキング表」は動作
- networkx / pyvis が入っていればインタラクティブ描画も可能
- 見せ方改善：KPIカード / 用語ヘルプ / 色付きランキング表 / 凡例
"""

from __future__ import annotations
import re
import itertools
import pandas as pd
import streamlit as st

# ---- 依存はオプショナル ----
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


# ====== 共有ユーティリティ（app.py側と独立に動くよう最小実装） ======
_AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")

def split_authors(cell) -> list[str]:
    if cell is None:
        return []
    return [w.strip() for w in _AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ====== データ加工 ======
@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame,
                         year_from: int | None = None,
                         year_to: int | None = None) -> pd.DataFrame:
    """
    入力: df（少なくとも '著者', '発行年' を含むこと）
    出力: edges DataFrame ['src', 'dst', 'weight']
    """
    use = df.copy()
    # 年で絞り込み（列が無いケースはそのまま）
    if "発行年" in use.columns and (year_from is not None and year_to is not None):
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= year_from) & (y <= year_to) | y.isna()]

    # 著者のペアを数える
    rows = []
    for a in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = split_authors(a)
        # 2名以上のときに同一論文内の全ペアを生成
        for s, t in itertools.combinations(sorted(set(names)), 2):
            rows.append((s, t))

    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = (edges.groupby("pair").size()
             .reset_index(name="weight"))
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]


def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree"):
    """
    edges: ['src','dst','weight']
    metric: 'degree'|'betweenness'|'eigenvector'
    戻り値: DataFrame ['author','centrality','note','raw_degree']
    """
    # networkx が無い場合：重み付き次数の簡易版
    if not HAS_NX:
        deg = pd.concat([
            edges.groupby("src")["weight"].sum(),
            edges.groupby("dst")["weight"].sum(),
        ], axis=1).fillna(0)
        deg["raw_degree"] = deg["weight"].sum(axis=1)  # 原始的な“つながりの強さ”
        out = deg["raw_degree"].sort_values(ascending=False).reset_index()
        out.columns = ["author", "raw_degree"]
        # 見やすさのため 0-1 正規化した擬似中心性も付与
        maxi = out["raw_degree"].max() or 1
        out["centrality"] = out["raw_degree"] / maxi
        out["note"] = "簡易次数（共著の強さの合計）/ networkxなし"
        return out[["author", "centrality", "note", "raw_degree"]]

    # networkx があるなら本格計算
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    # あわせて “生の次数（重み合計）” も出しておく
    raw_deg = {}
    for n in G.nodes():
        raw_deg[n] = sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n))

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
        note = "媒介中心性（橋渡し度合い）"
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
            note = "固有ベクトル中心性（影響力）"
        except Exception:
            cen = nx.degree_centrality(G)
            note = "固有ベクトル中心性→収束失敗のため次数中心性にフォールバック"
    else:
        cen = nx.degree_centrality(G)
        note = "次数中心性（つながりの多さ）"

    out = (pd.Series(cen, name="centrality")
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"index": "author"}))
    out["note"] = note
    out["raw_degree"] = out["author"].map(raw_deg).fillna(0).astype(float)
    return out[["author", "centrality", "note", "raw_degree"]]


def _draw_network(edges: pd.DataFrame,
                  top_nodes: list[str] | None = None,
                  min_weight: int = 1,
                  height_px: int = 650) -> None:
    """
    pyvis で描画（任意）。依存が無ければスキップ。
    """
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。表は利用できます。")
        return

    # サブグラフ（強いエッジのみ）
    edges_use = edges[edges["weight"] >= min_weight].copy()
    if edges_use.empty:
        st.warning("条件に合うエッジがありません。")
        return

    G = nx.Graph()
    for _, r in edges_use.iterrows():
        G.add_edge(r["src"], r["dst"], weight=int(r["weight"]))

    if top_nodes:
        keep = set(top_nodes)
        # トップ＋その隣接を残す
        keep |= {nbr for n in top_nodes for nbr in G.neighbors(n)}
        G = G.subgraph(keep).copy()

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.2, spring_length=110, spring_strength=0.02)
    for n in G.nodes():
        net.add_node(n, label=n)
    for s, t, d in G.edges(data=True):
        w = int(d.get("weight", 1))
        net.add_edge(s, t, value=w, title=f"共著回数: {w}")

    net.set_options("""
    {
      "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 40}},
      "edges": {"smooth": false}
    }
    """)
    net.show("coauthor_network.html")
    with open("coauthor_network.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=height_px, scrolling=True)


# ====== メインの描画関数 ======
def render_analysis_tab(df: pd.DataFrame) -> None:
    st.header("📊 分析：共著ネットワーク")

    # 年範囲
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        ymin, ymax = (int(y.min()), int(y.max())) if y.notna().any() else (1980, 2025)
    else:
        ymin, ymax = 1980, 2025

    # --- コントロール群
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        year_from, year_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    with c2:
        metric = st.selectbox(
            "中心性（分かりやすいラベルで表示）",
            ["degree", "betweenness", "eigenvector"],
            index=0,
            help="degree=つながりの多さ / betweenness=橋渡し役 / eigenvector=影響力"
        )
    with c3:
        top_n = st.number_input("ランキング件数", min_value=5, max_value=100, value=30, step=5)
    with c4:
        min_w = st.number_input("描画する最小共著回数 (w≥)", min_value=1, max_value=20, value=2, step=1)

    # --- エッジ作成
    edges = build_coauthor_edges(df, year_from, year_to)

    # --- KPI：まず“全体像”をカードで
    with st.container(border=True):
        cA, cB, cC, cD = st.columns(4)
        # 著者数・共著ペア数・総共著回数・平均共著回数
        authors_set = set()
        for a in df.get("著者", pd.Series(dtype=str)).fillna(""):
            authors_set.update(split_authors(a))
        unique_authors = len(authors_set)
        total_pairs = len(edges)
        total_w = int(edges["weight"].sum()) if not edges.empty else 0
        avg_w = round(edges["weight"].mean(), 2) if not edges.empty else 0.0

        cA.metric("著者数（ユニーク）", f"{unique_authors:,}")
        cB.metric("共著ペア数", f"{total_pairs:,}")
        cC.metric("総共著回数", f"{total_w:,}")
        cD.metric("1ペア平均共著回数", f"{avg_w}")

    st.markdown(
        "<div style='margin-top:.25rem; color:#555;'>"
        "※ まず全体像（誰がどれだけ一緒に書いているか）を把握 → 次にランキングとネットワークで詳細を見る"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("#### 中心性ランキング（＝影響・橋渡し・つながりの指標）")
    if edges.empty:
        st.info("共著エッジが見つかりませんでした。著者データを確認してください。")
        return

    # --- ランキング算出
    rank = _centrality_from_edges(edges, metric=metric).head(int(top_n)).copy()

    # わかりやすい表示名に揃える
    metric_label = {
        "degree": "次数中心性（つながりの多さ）",
        "betweenness": "媒介中心性（橋渡し度合い）",
        "eigenvector": "固有ベクトル中心性（影響力）",
    }[metric]

    # 小数丸め＆バー表示用の補助列
    rank["中心性(0-1)"] = rank["centrality"].astype(float).clip(lower=0, upper=1)
    rank["中心性(表示)"] = (rank["中心性(0-1)"] * 100).round(1)  # %
    rank = rank.rename(columns={
        "author": "著者",
        "raw_degree": "共著の強さ(合計)",
        "note": "指標の意味"
    })[["著者", "中心性(表示)", "共著の強さ(合計)", "指標の意味"]]

    # --- 見た目：棒バーで“直感”
    # pandas Styler を使ってバー表示（Streamlitは st.dataframe(df.style) に対応）
    sty = (rank.style
           .bar(subset=["中心性(表示)"], align="left", color=None)  # デフォ色（環境依存）
           .format({"中心性(表示)": "{:.1f}%"})
           .set_properties(**{"white-space": "nowrap"}))

    st.dataframe(sty, use_container_width=True, height=420)

    # 用語ミニヘルプ（ポップオーバーが無ければエクスパンダ）
    try:
        with st.popover("用語ヘルプ 🛈"):
            st.write(f"**{metric_label}** を表示中。")
            st.markdown(
                "- **次数中心性**：どれだけ多くの相手と結びついているか（コラボの多さ）\n"
                "- **媒介中心性**：クラスター間を“橋渡し”する度合い（情報の通り道）\n"
                "- **固有ベクトル中心性**：影響力の高い相手と繋がっているほど高評価\n"
                "- **共著の強さ(合計)**：重み（同じ相手との共著回数）を合算した実数指標"
            )
    except Exception:
        with st.expander("用語ヘルプ 🛈", expanded=False):
            st.write(f"**{metric_label}** を表示中。")
            st.markdown(
                "- **次数中心性**：どれだけ多くの相手と結びついているか（コラボの多さ）\n"
                "- **媒介中心性**：クラスター間を“橋渡し”する度合い（情報の通り道）\n"
                "- **固有ベクトル中心性**：影響力の高い相手と繋がっているほど高評価\n"
                "- **共著の強さ(合計)**：重み（同じ相手との共著回数）を合算した実数指標"
            )

    # --- ネットワーク（任意）
    with st.expander("🌐 全体ネットワーク（任意・依存あり）", expanded=False):
        st.caption("※ networkx / pyvis がインストールされている場合のみ描画します。"
                   " ノード＝著者、エッジの太さ＝共著回数。")
        top_only = st.toggle("トップNの周辺だけ可視化（軽量表示）", value=True)
        top_nodes = rank["著者"].tolist() if top_only else None
        draw = st.button("ネットワークを描画する")
        if draw:
            _draw_network(edges, top_nodes=top_nodes, min_weight=int(min_w), height_px=700)

    # --- ちいさな凡例
    st.caption("凡例：バー＝中心性(0-1)を％表示 / 『共著の強さ(合計)』＝同じ相手との共著回数の合計（重み付き）")