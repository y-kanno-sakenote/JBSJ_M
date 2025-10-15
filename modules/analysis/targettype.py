# modules/analysis/targettype.py
# -*- coding: utf-8 -*-
"""
対象物・研究タイプ 分析タブ（完成版）
- ① 構成比・クロス集計：対象物 / 研究タイプの件数、クロスヒートマップ
- ② 経年トレンド：年ごとの件数推移、対象の比較、移動平均オプション
- ③ 共起ネットワーク：同一論文内の共起（対象物 / 研究タイプ / 両方）をネットワークで可視化
  * 重い処理はディスクキャッシュ（modules/common/cache_utils.py）で永続化
"""

from __future__ import annotations
import re
import itertools
from typing import List, Tuple, Dict, Set

import pandas as pd
import streamlit as st

# 共通フィルタバー
from modules.common.filters import render_filter_bar
def _df_from_filter_result(res, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(res, pd.DataFrame):
        return res
    try:
        if isinstance(res, (list, tuple)) and len(res) > 0:
            x0 = res[0]
            if isinstance(x0, pd.DataFrame):
                return x0
        if isinstance(res, dict):
            for k in ("df", "df_use", "filtered_df"):
                v = res.get(k)
                if isinstance(v, pd.DataFrame):
                    return v
    except Exception:
        pass
    return fallback_df

# 表示順（固定）
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

def _order_options(all_options: list[str], preferred: list[str]) -> list[str]:
    """preferredにあるものはその順で先頭、それ以外は五十音/アルファベット順で後ろへ"""
    s = set(all_options)
    head = [x for x in preferred if x in s]
    tail = sorted([x for x in all_options if x not in preferred])
    return head + tail

# --- Optional deps (なくても動く) ---
try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

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

# --- 永続キャッシュIO ---
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========== ユーティリティ ==========
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    """列の文字列に needles のいずれかが部分一致（小文字・空白正規化）"""
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)

@st.cache_data(ttl=600, show_spinner=False)
def _year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

def _apply_filters(df: pd.DataFrame,
                   y_from: int, y_to: int,
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


# ========= ① 構成比・クロス集計 =========
@st.cache_data(ttl=600, show_spinner=False)
def _count_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    bags: List[str] = []
    for v in df[col].fillna(""):
        bags += split_multi(v)
    if not bags:
        return pd.Series(dtype=int)
    s = pd.Series(bags)
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def _cross_counts(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """A×Bのクロス件数（同一論文内で全組合せをカウント）"""
    if col_a not in df.columns or col_b not in df.columns:
        return pd.DataFrame(columns=["A", "B", "count"])
    rows = []
    for _, r in df.iterrows():
        As = list(dict.fromkeys(split_multi(r.get(col_a, ""))))
        Bs = list(dict.fromkeys(split_multi(r.get(col_b, ""))))
        for a in As:
            for b in Bs:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["A", "B", "count"])
    c = pd.DataFrame(rows, columns=["A", "B"]).value_counts().reset_index(name="count")
    return c.sort_values("count", ascending=False).reset_index(drop=True)


def _render_distribution_block(df: pd.DataFrame) -> None:
    # Small subheading style for inline subttls
    st.markdown("<style>.subttl{font-size:0.95rem; opacity:0.75; margin:0 0 0.25rem;}</style>", unsafe_allow_html=True)

    # ---- 対象物集計 ----
    tg_series = (
        df.get("対象物_top3", pd.Series(dtype=str))
          .fillna("")
          .apply(lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()])
    )
    tg_flat = [w for lst in tg_series for w in lst]
    tg_counts = pd.Series(tg_flat, dtype="object").value_counts()
    tg_df = tg_counts.reset_index()
    tg_df.columns = ["対象物", "件数"]
    tg_df = tg_df.sort_values("件数", ascending=False)

    # ---- 研究タイプ集計 ----
    tp_series = (
        df.get("研究タイプ_top3", pd.Series(dtype=str))
          .fillna("")
          .apply(lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()])
    )
    tp_flat = [w for lst in tp_series for w in lst]
    tp_counts = pd.Series(tp_flat, dtype="object").value_counts()
    tp_df = tp_counts.reset_index()
    tp_df.columns = ["研究タイプ", "件数"]
    tp_df = tp_df.sort_values("件数", ascending=False)

    # 合計件数を計算
    tg_total = int(tg_df["件数"].sum()) if not tg_df.empty else 0
    tp_total = int(tp_df["件数"].sum()) if not tp_df.empty else 0

    if tg_df.empty and tp_df.empty:
        st.info("該当データがありません。フィルタを調整してください。")
        return

    c1, c2 = st.columns(2)
    with c1:
        if tg_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            try:
                import plotly.express as px  # 遅延import
                fig = px.bar(
                    tg_df,
                    x="対象物",
                    y="件数",
                    text_auto=True,
                    title=f"対象物の出現件数（合計: {tg_total:,}件）",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, yaxis_title="件数")
                fig.update_xaxes(tickangle=45, automargin=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(tg_df.set_index("対象物")["件数"])

    with c2:
        if tp_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            try:
                import plotly.express as px
                fig2 = px.bar(
                    tp_df,
                    x="研究タイプ",
                    y="件数",
                    text_auto=True,
                    title=f"研究タイプの出現件数（合計: {tp_total:,}件）",
                )
                fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, yaxis_title="件数")
                fig2.update_xaxes(tickangle=45, automargin=True)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.bar_chart(tp_df.set_index("研究タイプ")["件数"])


def _render_cross_block(df: pd.DataFrame) -> None:
    # Use the same subttl style for the cross heatmap
    st.markdown('<div style="font-weight=600; font-size:1.1rem; margin:0 0 0.25rem;">対象物 × 研究タイプ（クロスヒートマップ）</div>', unsafe_allow_html=True)

    cross = _cross_counts(df, "対象物_top3", "研究タイプ_top3")
    if cross.empty:
        st.info("クロス集計できるデータがありません。")
        return

    # ピボット（行=研究タイプ、列=対象物）
    piv = cross.pivot(index="B", columns="A", values="count").fillna(0).astype(int)
    piv.index.name = "研究タイプ"
    piv.columns.name = "対象物"

    # 並び順を固定（指定順 → 未定義カテゴリは後尾で五十音/アルファベット順）
    cols_order = [x for x in TARGET_ORDER if x in piv.columns] + sorted([x for x in piv.columns if x not in TARGET_ORDER])
    idx_order  = [x for x in TYPE_ORDER   if x in piv.index  ] + sorted([x for x in piv.index  if x not in TYPE_ORDER])
    piv = piv.reindex(index=idx_order, columns=cols_order)

    # 下部に配置するチェックボックスの現在値をセッションから参照（初期は False）
    show_values = bool(st.session_state.get("obj_cross_show_values", False))

    if HAS_PX:
        import plotly.express as px
        fig = px.imshow(
            piv,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(color="件数"),
        )
        fig.update_xaxes(categoryorder="array", categoryarray=cols_order, tickangle=45, automargin=True)
        fig.update_yaxes(categoryorder="array", categoryarray=idx_order, automargin=True)
        # 値表示とホバーの明確化（セル値はトグルでON/OFF）
        if show_values:
            try:
                fig.update_traces(
                    text=piv.values,
                    texttemplate="%{text}",
                    hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>"
                )
            except Exception:
                # 古いPlotlyでもホバーは維持
                fig.update_traces(hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>")
        else:
            fig.update_traces(hovertemplate="研究タイプ=%{y}<br>対象物=%{x}<br>件数=%{z}<extra></extra>")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), coloraxis_colorbar_title="件数")
        st.plotly_chart(fig, use_container_width=True)
        # 右下に配置するセル値表示トグル
        rb_spacer, rb_cb = st.columns([6, 1])
        with rb_cb:
            st.checkbox(
                "セルの値を表示",
                value=show_values,
                key="obj_cross_show_values",
                help="ヒートマップの各セルに件数を直接表示します。表示すると読みやすくなる一方、カテゴリ数が多い場合は見づらくなることがあります。"
            )
    else:
        st.dataframe(piv, use_container_width=True)
        # 右下トグル（データフレーム表示時も配置のみ実施）
        rb_spacer, rb_cb = st.columns([6, 1])
        with rb_cb:
            st.checkbox(
                "セルの値を表示",
                value=show_values,
                key="obj_cross_show_values",
                help="ヒートマップの各セルに件数を直接表示します。表示すると読みやすくなる一方、カテゴリ数が多い場合は見づらくなることがあります。"
            )

# ========= ② 経年トレンド =========
@st.cache_data(ttl=600, show_spinner=False)
def _yearly_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """年×項目の件数（同一論文内の重複は1件としてカウント）"""
    if col not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", col, "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        items = list(dict.fromkeys(split_multi(r.get(col, ""))))
        for it in items:
            rows.append((int(y), it))
    if not rows:
        return pd.DataFrame(columns=["発行年", col, "count"])
    c = pd.DataFrame(rows, columns=["発行年", col]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年", "count"], ascending=[True, False]).reset_index(drop=True)

def _render_trend_block(df: pd.DataFrame) -> None:
    # 1行（2:6:2）に「対象」「表示する項目」「移動平均」を配置
    c1, c2, c3 = st.columns([1.5, 8, 1.5])

    # 対象（左）
    with c1:
        target_mode = st.selectbox(
            "対象",
            ["対象物_top3", "研究タイプ_top3"],
            index=0,
            key="obj_trend_mode",
            # 表示のみ「対象物」「研究タイプ」にする（内部値は *_top3 を維持）
            format_func=lambda x: "対象物" if x == "対象物_top3" else ("研究タイプ" if x == "研究タイプ_top3" else str(x))
        )

    use = df  # 既にフィルタ済み

    # 候補抽出と順序固定（中央の multiselect で使う）
    all_items_raw = sorted({
        t for v in use.get(target_mode, pd.Series(dtype=str)).fillna("")
        for t in split_multi(v)
    })
    if target_mode == "対象物_top3":
        all_items = _order_options(all_items_raw, TARGET_ORDER)
    else:
        all_items = _order_options(all_items_raw, TYPE_ORDER)

    # 表示する項目（中央）
    with c2:
        sel = st.multiselect(
            "表示する項目（複数可）",
            options=all_items[:1000],
            default=all_items[: min(0, len(all_items))],  # 既存仕様：初期は空
            key="obj_trend_items",
        )

    # 移動平均（右）
    with c3:
        ma = st.number_input(
            "移動平均（年）",
            min_value=1, max_value=7, value=1, step=1,
            key="obj_trend_ma"
        )

    yearly = _yearly_counts(use, target_mode)
    if yearly.empty:
        st.info("データがありません。")
        return

    piv = yearly.pivot_table(index="発行年", columns=target_mode, values="count", aggfunc="sum").fillna(0).sort_index()
    if sel:
        piv = piv[[c for c in sel if c in piv.columns]]

    if piv.shape[1] == 0:
        st.info("表示対象がありません。左のリストから1つ以上選んでください。")
        return

    if ma > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    _sel_key = ",".join(sel) if sel else "__ALL__"
    _uniq_key = f"obj_trend_plot|{target_mode}|{_sel_key}|ma{ma}"

    if target_mode == "対象物_top3":
        legend_order = [x for x in TARGET_ORDER if x in piv.columns]
    else:
        legend_order = [x for x in TYPE_ORDER if x in piv.columns]

    if HAS_PX:
        fig = px.line(
            piv.reset_index().melt(id_vars="発行年", var_name="項目", value_name="件数"),
            x="発行年", y="件数", color="項目", markers=True,
            category_orders={"項目": legend_order}
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
    else:
        st.line_chart(piv, key=_uniq_key)

# ========= ③ 共起ネットワーク =========
def _build_cooccur_edges(df: pd.DataFrame,
                         mode: str,
                         min_edge: int) -> pd.DataFrame:
    """
    mode: '対象物のみ' | '研究タイプのみ' | '対象物×研究タイプ'
    戻り値: ['src','dst','weight']
    """
    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        tg = list(dict.fromkeys(split_multi(r.get("対象物_top3", ""))))
        tp = list(dict.fromkeys(split_multi(r.get("研究タイプ_top3", ""))))
        if mode == "対象物のみ":
            items = tg
            pairs = itertools.combinations(sorted(items), 2)
        elif mode == "研究タイプのみ":
            items = tp
            pairs = itertools.combinations(sorted(items), 2)
        else:  # 対象物×研究タイプ（双部）
            pairs = itertools.product(sorted(set(tg)), sorted(set(tp)))
        for a, b in pairs:
            if a and b and a != b:
                rows.append((a, b))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。")
        return
    if edges.empty:
        st.warning("エッジがありません。")
        return

    import math
    import networkx as nx
    from pyvis.network import Network

    # 1) NetworkX Graph（weightつき）
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    # 2) ノード強度（エッジ重みの総和）＝重要度の素点
    strength = {}
    for n in G.nodes():
        strength[n] = sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True))

    # 3) コミュニティ検出（色分け用）
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_id = {}
        for i, cset in enumerate(comms):
            for n in cset:
                comm_id[n] = i
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}

    # 4) ラベルを出すノード（強い順に上位だけにラベル）
    #    数は自動（最大でも40）。小さいネットなら全ラベル。
    max_labels = 40
    sorted_nodes = sorted(G.nodes(), key=lambda n: strength.get(n, 0), reverse=True)
    label_set = set(sorted_nodes[: max_labels if len(sorted_nodes) > max_labels else len(sorted_nodes)])

    # 5) PyVis へ
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    # 物理＆相互作用を調整：重なり軽減・ホバー強化
    net.set_options("""
    {
      "interaction": { "hover": true, "tooltipDelay": 200, "zoomView": true, "dragView": true },
      "physics": {
        "stabilization": { "enabled": true, "iterations": 200 },
        "barnesHut": { "gravitationalConstant": -25000, "centralGravity": 0.2, "springLength": 140, "springConstant": 0.025, "damping": 0.4, "avoidOverlap": 0.5 }
      },
      "nodes": { "shape": "dot" },
      "edges": { "smooth": { "type": "dynamic" } }
    }
    """)

    # ノード追加（サイズ=logスケール、色=コミュニティ、ラベルは上位のみ）
    # 目安: size 6〜28
    def size_for(n):
        s = max(1.0, float(strength.get(n, 1)))
        return max(6.0, min(28.0, 6.0 + 4.0 * math.log1p(s)))

    for n in G.nodes():
        lbl = n if n in label_set else ""   # テキストラベルは上位だけ
        title = f"{n}<br>総共起重み: {strength.get(n,0):,.0f}"  # ホバーで全ノード名を見せる
        net.add_node(
            n,
            label=lbl,
            title=title,
            value=strength.get(n, 0),
            size=size_for(n),
            group=int(comm_id.get(n, 0)),
        )

    # エッジ追加（太さ=logスケール）
    def width_for(w):
        return max(1.0, min(10.0, 1.0 + 2.0 * math.log1p(float(w))))
    for s, t, d in G.edges(data=True):
        w = d.get("weight", 1)
        net.add_edge(s, t, value=float(w), width=width_for(w), title=f"共起: {int(w)} 回")

    # 6) 生成→埋め込み
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)

    # 7) ダウンロード（既存のまま）
    st.download_button(
        "📥 ネットワークHTML",
        data=html.encode("utf-8"),
        file_name="cooccurrence_network.html",
        mime="text/html",
        key="dl_pyvis_html",
        help="このネットワークを単独のHTMLファイルとして保存します（ブラウザでそのまま開けます）。"
    )
    
def _render_cooccurrence_block(df_use: pd.DataFrame) -> None:
    # 年・対象物・研究タイプのフィルタUIは除外し、ネットワーク種別・しきい値・ノード数のみ
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        mode = st.selectbox("ネットワークの種類", ["対象物のみ", "研究タイプのみ", "対象物×研究タイプ"], index=0, key="obj_net_mode")
    with c2:
        min_edge = st.number_input("最低共起数（同時出現）", min_value=1, max_value=50, value=3, step=1, key="obj_net_minw")
    with c3:
        topN = st.number_input("表示するノード数（多い順）", min_value=30, max_value=300, value=120, step=10, key="obj_net_topn")

    use = df_use
    edges = _build_cooccur_edges(use, mode, int(min_edge))
    if not edges.empty and int(topN) > 0:
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    st.caption(f"エッジ数: {len(edges)}")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)

    # 2) ネットワーク描画
    with st.expander("🕸️ ネットワークを可視化", expanded=False):
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="obj_net_draw"):
                _draw_pyvis_from_edges(edges, height_px=680)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")


# ========= エクスポート：タブ本体 =========
def render_targettype_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
          <h2 style="margin:0;">🧬 対象物・研究タイプ分析</h2>
          <span style="opacity:0.8;">対象物・研究タイプの分布・クロス集計・共起ネットワーク・トレンドを確認できます。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 共通フィルタバー
    _flt_res = render_filter_bar(
        df,
        key_prefix="obj",
        target_order=TARGET_ORDER,
        type_order=TYPE_ORDER,
    )
    df_use = _df_from_filter_result(_flt_res, df)

    tab1, tab2, tab3 = st.tabs([
        "① 構成比・クロス集計",
        "② 共起ネットワーク",
        "③ トレンド分析",
    ])

    with tab1:
        # 上段：対象物/研究タイプの並列バー（共通フィルタ適用済みの df_use をそのまま使用）
        _render_distribution_block(df_use)
        st.divider()
        _render_cross_block(df_use)

    with tab2:
        _render_cooccurrence_block(df_use)

    with tab3:
        _render_trend_block(df_use)