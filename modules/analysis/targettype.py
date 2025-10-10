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
    st.markdown("### 📊 分布：対象物／研究タイプ の件数")

    # ---- 対象物集計 ----
    tg_series = (
        df.get("対象物_top3", pd.Series(dtype=str))
          .fillna("")
          .apply(lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()])
    )
    tg_flat = [w for lst in tg_series for w in lst]
    tg_counts = pd.Series(tg_flat, dtype="object").value_counts()

    if tg_counts.empty:
        st.info("対象物の集計対象データがありません。")
    else:
        tg_df = tg_counts.reset_index()
        tg_df.columns = ["対象物", "件数"]
        tg_df = tg_df.sort_values("件数", ascending=False)
        top_n = st.number_input("対象物の表示件数", min_value=5, max_value=100, value=20, step=5, key="tg_topn")
        tg_df_top = tg_df.head(int(top_n))

        try:
            import plotly.express as px  # 遅延import
            fig = px.bar(
                tg_df_top,
                x="対象物",
                y="件数",
                text_auto=True,
                title="対象物の出現件数（上位）",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.bar_chart(tg_df_top.set_index("対象物")["件数"])

    st.divider()

    # ---- 研究タイプ集計 ----
    tp_series = (
        df.get("研究タイプ_top3", pd.Series(dtype=str))
          .fillna("")
          .apply(lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()])
    )
    tp_flat = [w for lst in tp_series for w in lst]
    tp_counts = pd.Series(tp_flat, dtype="object").value_counts()

    if tp_counts.empty:
        st.info("研究タイプの集計対象データがありません。")
    else:
        tp_df = tp_counts.reset_index()
        tp_df.columns = ["研究タイプ", "件数"]
        tp_df = tp_df.sort_values("件数", ascending=False)
        top_n_tp = st.number_input("研究タイプの表示件数", min_value=5, max_value=100, value=20, step=5, key="tp_topn")
        tp_df_top = tp_df.head(int(top_n_tp))

        try:
            import plotly.express as px
            fig2 = px.bar(
                tp_df_top,
                x="研究タイプ",
                y="件数",
                text_auto=True,
                title="研究タイプの出現件数（上位）",
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.bar_chart(tp_df_top.set_index("研究タイプ")["件数"])


# ---- ①-2 追加：対象物×研究タイプのクロスヒートマップ（年範囲＋最小件数フィルタ付き）----
def _render_cross_block(df: pd.DataFrame) -> None:
    st.markdown("### ①-2 対象物 × 研究タイプ（クロスヒートマップ）")

    ymin_all, ymax_all = _year_min_max(df)
    c1, c2 = st.columns([2, 1])
    with c1:
        y_from, y_to = st.slider(
            "対象年（範囲）",
            min_value=ymin_all, max_value=ymax_all,
            value=(ymin_all, ymax_all),
            key="obj_cross_year",
        )
    with c2:
        min_cnt = st.number_input("表示する最小件数 (≥)", min_value=1, max_value=50, value=3, step=1, key="obj_cross_min")

    # 年フィルタだけ軽量に適用
    use = _apply_filters(df, y_from, y_to, [], [])

    cross = _cross_counts(use, "対象物_top3", "研究タイプ_top3")
    if cross.empty:
        st.info("クロス集計できるデータがありません。")
        return

    # 最小件数でフィルタ
    cross = cross[cross["count"] >= int(min_cnt)].copy()
    if cross.empty:
        st.info("この閾値ではデータがありません。閾値を下げてください。")
        return

    # ピボット（行=研究タイプ、列=対象物）
    piv = cross.pivot(index="B", columns="A", values="count").fillna(0).astype(int)
    piv.index.name = "研究タイプ"
    piv.columns.name = "対象物"

    if HAS_PX:
        fig = px.imshow(
            piv,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(color="件数"),
        )
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(piv, use_container_width=True)


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
    st.markdown("### ② 経年変化（トレンド）")

    ymin_all, ymax_all = _year_min_max(df)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin_all, max_value=ymax_all, value=(ymin_all, ymax_all), key="obj_trend_year")
    with c2:
        target_mode = st.selectbox("対象", ["対象物_top3","研究タイプ_top3"], index=0, key="obj_trend_mode")
    with c3:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="obj_trend_ma")

    # 候補と選択
    # 候補と選択
    use = _apply_filters(df, y_from, y_to, [], [])

    # 生の候補を抽出
    all_items_raw = sorted({
        t for v in use.get(target_mode, pd.Series(dtype=str)).fillna("")
        for t in split_multi(v)
    })

    # ★ 表示順を固定（対象物/研究タイプで順序を切替）
    if target_mode == "対象物_top3":
        all_items = _order_options(all_items_raw, TARGET_ORDER)
    else:  # "研究タイプ_top3"
        all_items = _order_options(all_items_raw, TYPE_ORDER)

    # multiselect（表示順そのまま、デフォルト選択も同順で上位から）
    sel = st.multiselect(
        "表示する項目（複数可）",
        all_items[:1000],
        default=all_items[: min(0, len(all_items))],
        key="obj_trend_items",
    )

    yearly = _yearly_counts(use, target_mode)
    if yearly.empty:
        st.info("データがありません。")
        return

    piv = yearly.pivot_table(index="発行年", columns=target_mode, values="count", aggfunc="sum").fillna(0).sort_index()
    if sel:
        piv = piv[[c for c in sel if c in piv.columns]]

    # ★ 空列（全解除や不一致）なら描画せずに案内して終了
    if piv.shape[1] == 0:
        st.info("表示対象がありません。左のリストから1つ以上選んでください。")
        return

    if ma > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    # ★ 一意キー：年範囲・モード・選択・MAでユニーク化
    _sel_key = ",".join(sel) if sel else "__ALL__"
    _uniq_key = f"obj_trend_plot|{y_from}-{y_to}|{target_mode}|{_sel_key}|ma{ma}"

    if HAS_PX:
        fig = px.line(
            piv.reset_index().melt(id_vars="発行年", var_name="項目", value_name="件数"),
            x="発行年", y="件数", color="項目", markers=True
        )
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        # ★ key を付けて重複ID回避（旧版でも key は利用可能）
        st.plotly_chart(fig, use_container_width=True, key=_uniq_key)
    else:
        # st.line_chart も key を付けておくと安心
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

    # NX Graph
    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = r["src"], r["dst"], int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    # PyVis
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=120, spring_strength=0.02)
    net.from_nx(G)
    # 生成→埋め込み（open_browserを使わない）
    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)

def _render_cooccurrence_block(df: pd.DataFrame) -> None:
    st.markdown("### ③ 共起ネットワーク（対象物・研究タイプ）")

    ymin_all, ymax_all = _year_min_max(df)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin_all, max_value=ymax_all, value=(ymin_all, ymax_all), key="obj_net_year")
    with c2:
        mode = st.selectbox("ネットワークの種類", ["対象物のみ", "研究タイプのみ", "対象物×研究タイプ"], index=0, key="obj_net_mode")
    with c3:
        min_edge = st.number_input("エッジ最小回数 (w≥)", min_value=1, max_value=50, value=3, step=1, key="obj_net_minw")
    with c4:
        topN = st.number_input("上位のノード数（上限）", min_value=30, max_value=300, value=120, step=10, key="obj_net_topn")

    # 年だけ当てたデータから候補を抽出
    use_year = _apply_filters(df, y_from, y_to, [], [])
    tg_all = sorted({t for v in use_year.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                    for t in split_multi(v)})
    tp_all = sorted({t for v in use_year.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                    for t in split_multi(v)})

    # ★ 表示順を固定
    tg_all = _order_options(tg_all, TARGET_ORDER)
    tp_all = _order_options(tp_all, TYPE_ORDER)

    c5, c6 = st.columns([1, 1])
    with c5:
        tg_needles = st.multiselect("対象物で絞り込み（選択）", tg_all, default=[], key="obj_net_tg_sel")
    with c6:
        tp_needles = st.multiselect("研究タイプで絞り込み（選択）", tp_all, default=[], key="obj_net_tp_sel")

    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)

    # ---- キャッシュキー ----
    cache_key = f"objnet|{y_from}-{y_to}|{mode}|min{min_edge}|top{topN}|tg{','.join(tg_needles)}|tp{','.join(tp_needles)}"

    # 1) エッジ構築（重いので永続キャッシュ）
    edges = None
    if HAS_DISK_CACHE:
        path_edges = cache_csv_path("obj_net_edges", cache_key)
        cached = load_csv_if_exists(path_edges)
        if cached is not None:
            edges = cached

    if edges is None:
        edges = _build_cooccur_edges(use, mode, int(min_edge))
        # 上位ノード制限：出現多いノードを残す
        if not edges.empty and int(topN) > 0:
            deg = pd.concat([edges.groupby("src")["weight"].sum(),
                             edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
            keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
            edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)
        if HAS_DISK_CACHE:
            save_csv(edges, path_edges)

    st.caption(f"エッジ数: {len(edges)}")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)

    # 2) ネットワーク描画
    with st.expander("🕸️ ネットワークを描画（PyVis / 任意依存）", expanded=False):
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="obj_net_draw"):
                _draw_pyvis_from_edges(edges, height_px=680)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")


# ========= エクスポート：タブ本体 =========
def render_targettype_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧂 対象物・研究タイプ分析")

    tab1, tab2, tab3 = st.tabs([
        "① 構成比・クロス集計",
        "② 経年変化",
        "③ 共起ネットワーク",
    ])

    with tab1:
        _render_distribution_block(df)
        st.divider()
        _render_cross_block(df)   # ← 追加

    with tab2:
        _render_trend_block(df)

    with tab3:
        _render_cooccurrence_block(df)