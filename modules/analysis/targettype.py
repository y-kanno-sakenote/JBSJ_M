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


# ==== 新規ユーティリティ ====

def _split_and_count_series(df: pd.DataFrame, col: str, x_name: str, y_name: str) -> pd.DataFrame:
    """
    指定列（; / ・空白区切り）を分解→頻度集計→降順のDataFrameを返す。
    戻り: [x_name, y_name]
    """
    if col not in df.columns:
        return pd.DataFrame(columns=[x_name, y_name])
    series = (
        df.get(col, pd.Series(dtype=str))
          .fillna("")
          .apply(lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()])
    )
    flat = [w for lst in series for w in lst]
    if not flat:
        return pd.DataFrame(columns=[x_name, y_name])
    s = pd.Series(flat, dtype="object").value_counts()
    out = s.reset_index()
    out.columns = [x_name, y_name]
    return out.sort_values(y_name, ascending=False)


def _px_bar_count(df_xy: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Plotlyが使える場合に件数バーを統一スタイルで返す（なければNone）。
    """
    if not HAS_PX:
        return None
    try:
        import plotly.express as px  # 遅延import
        fig = px.bar(
            df_xy,
            x=x_col,
            y=y_col,
            text_auto=True,
            title=title,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, yaxis_title=y_col)
        fig.update_xaxes(tickangle=45, automargin=True)
        return fig
    except Exception:
        return None


def _ordered_index_and_columns(piv: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    ヒートマップ用に index/columns の表示順を TARGET_ORDER / TYPE_ORDER で整列し、
    未定義カテゴリは末尾で五十音（アルファベット）順にする。
    行=研究タイプ、列=対象物 を想定。
    戻り: (idx_order, cols_order)
    """
    cols = list(piv.columns)
    idxs  = list(piv.index)
    cols_order = [x for x in TARGET_ORDER if x in cols] + sorted([x for x in cols if x not in TARGET_ORDER])
    idx_order  = [x for x in TYPE_ORDER   if x in idxs] + sorted([x for x in idxs if x not in TYPE_ORDER])
    return idx_order, cols_order


def _node_options_for_mode(df_use: pd.DataFrame, mode: str) -> list[str]:
    """
    ネットワーク種類に応じたノード候補（指定順で並び替え）。
    """
    if mode == "対象物のみ":
        cand = sorted({t for v in df_use.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        return _order_options(cand, TARGET_ORDER)
    elif mode == "研究タイプのみ":
        cand = sorted({t for v in df_use.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        return _order_options(cand, TYPE_ORDER)
    else:
        cand_tg = sorted({t for v in df_use.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        cand_tp = sorted({t for v in df_use.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)})
        cand_tg = _order_options(cand_tg, TARGET_ORDER)
        cand_tp = _order_options(cand_tp, TYPE_ORDER)
        return cand_tg + [x for x in cand_tp if x not in cand_tg]

# === 追加: タイトル列選好・クラスタ検出・例示・色四角 ===
def _prefer_title_column(df: pd.DataFrame) -> str | None:
    """候補の中からタイトル列名を返す。無ければ None。"""
    for c in ["タイトル", "論文タイトル", "title", "Title", "題名"]:
        if c in df.columns:
            return c
    return None

def _compute_communities_from_edges(edges: pd.DataFrame):
    """
    エッジから networkx.Graph を作り、コミュニティID（0,1,2,...) を返す。
    あわせて可視化で使う固定パレット（クラスタID→色 hex）も返す。
    戻り値: (comm_id: dict[node,int], palette: dict[int,str])
    """
    if edges.empty or not HAS_NX:
        return {}, {}
    import networkx as nx
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1)))
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_id = {}
        for i, cset in enumerate(comms):
            for n in cset:
                comm_id[str(n)] = i
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}

    base_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
        "#3182bd","#e6550d","#31a354","#756bb1","#636363",
        "#9ecae1","#fdae6b","#74c476","#bcbddc","#bdbdbd",
    ]
    palette = {i: base_colors[i % len(base_colors)] for i in set(comm_id.values())}
    return comm_id, palette


def _example_titles_for_edge(df: pd.DataFrame, mode: str, a: str, b: str, limit: int = 3) -> list[str]:
    """
    エッジ (a,b) に対応する論文タイトルの例（最大 limit 件）を抽出。
    mode に応じて同一列/別列を判定する。
    """
    title_col = _prefer_title_column(df)
    if title_col is None:
        return []

    tg_col = "対象物_top3"; tp_col = "研究タイプ_top3"
    def _tokset(val: str) -> set[str]:
        return set(split_multi(val))

    rows = []
    for _, r in df.iterrows():
        tg = _tokset(r.get(tg_col, "")) if tg_col in df.columns else set()
        tp = _tokset(r.get(tp_col, "")) if tp_col in df.columns else set()
        ok = False
        if mode == "対象物のみ":
            ok = (a in tg) and (b in tg)
        elif mode == "研究タイプのみ":
            ok = (a in tp) and (b in tp)
        else:
            ok = (a in tg) and (b in tp)
        if ok:
            rows.append((r.get("発行年", None), str(r.get(title_col, ""))))
    if not rows:
        return []
    try:
        rows = sorted(rows, key=lambda x: (pd.to_numeric(x[0], errors="coerce") if x[0] is not None else -1), reverse=True)
    except Exception:
        pass
    return [t for _, t in rows[:limit]]



def _color_square_html(color_hex: str, size_px: int = 12) -> str:
    """小さな色四角のHTML（DataFrameのHTML表示で使う）。"""
    s = f"display:inline-block;width:{size_px}px;height:{size_px}px;border-radius:2px;background:{color_hex};border:1px solid #999;margin-right:6px;"
    return f'<span style="{s}"></span>'


def _color_square_data_uri(color_hex: str, size_px: int = 12) -> str:
    """Return a data URI for a tiny colored square suitable for st.dataframe ImageColumn.
    Prefer PNG(base64) because some Streamlit builds do not render inline SVG data URIs."""
    color = str(color_hex or "#999999")
    size = max(6, int(size_px))
    try:
        from PIL import Image
        import io, base64
        img = Image.new("RGBA", (size, size), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # Fallback: SVG as base64 (safer than utf8 inline)
        import base64
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"><rect width="{size}" height="{size}" fill="{color}"/></svg>'
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return "data:image/svg+xml;base64," + b64


# クラスタ凡例（色+件数）を描画するヘルパ
def _render_cluster_legend_counts(palette: dict[int, str], comm_id: dict[str, int]) -> None:
    """
    クラスタ凡例（例: 「クラスタ凡例  ⬛ C1（49語） ⬛ C2（39語） ...」）
    - palette: {cluster_id -> color_hex}
    - comm_id: {node -> cluster_id}
    """
    if not palette:
        return
    # clusterごとのユニークノード数を集計
    counts: dict[int, int] = {}
    for n, cid in (comm_id or {}).items():
        counts[cid] = counts.get(cid, 0) + 1

    # 表示順は cluster_id の昇順
    items = sorted(palette.items(), key=lambda kv: kv[0])

    # HTMLを生成（インライン・柔らかい余白）
    html = ['<div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:6px 0 10px 0;">']
    html.append('<span style="font-weight:700; margin-right:4px;">クラスター凡例</span>')
    for cid, color in items:
        cnt = counts.get(cid, 0)
        square = f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{color};border:1px solid #999;margin:0 6px 0 2px;"></span>'
        # 表示は 1始まり: C1, C2, ...
        label = f'C{int(cid)+1}（{cnt}語）'
        html.append(f'<span style="display:inline-flex;align-items:center;gap:4px;">{square}<span style="font-size:12.5px;opacity:0.9;">{label}</span></span>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_distribution_block(df: pd.DataFrame) -> None:
    # Small subheading style for inline subttls
    st.markdown("<style>.subttl{font-size:0.95rem; opacity:0.75; margin:0 0 0.25rem;}</style>", unsafe_allow_html=True)

    # ---- 対象物集計 ----
    tg_df = _split_and_count_series(df, "対象物_top3", "対象物", "件数")
    tg_total = int(tg_df["件数"].sum()) if not tg_df.empty else 0

    # ---- 研究タイプ集計 ----
    tp_df = _split_and_count_series(df, "研究タイプ_top3", "研究タイプ", "件数")
    tp_total = int(tp_df["件数"].sum()) if not tp_df.empty else 0

    if tg_df.empty and tp_df.empty:
        st.info("該当データがありません。フィルタを調整してください。")
        return

    c1, c2 = st.columns(2)
    with c1:
        if tg_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            fig = _px_bar_count(tg_df, "対象物", "件数", f"対象物の出現件数（合計: {tg_total:,}件）")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(tg_df.set_index("対象物")["件数"])

    with c2:
        if tp_df.empty:
            st.info("該当データがありません。フィルタを調整してください。")
        else:
            fig2 = _px_bar_count(tp_df, "研究タイプ", "件数", f"研究タイプの出現件数（合計: {tp_total:,}件）")
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
            else:
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
    idx_order, cols_order = _ordered_index_and_columns(piv)
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
    # 1行に「対象」「最新年Top5自動」「表示する項目」「移動平均」を配置
    c1, c2, c3, c4 = st.columns([1.5, 1.6, 6.6, 1.5])

    # 対象（左）
    with c1:
        target_mode = st.selectbox(
            "対象",
            ["対象物_top3", "研究タイプ_top3"],
            index=0,
            key="obj_trend_mode",
            # 表示ラベルのみ変更（内部値は *_top3）
            format_func=lambda x: "対象物" if x == "対象物_top3" else ("研究タイプ" if x == "研究タイプ_top3" else str(x))
        )

    use = df  # 既にフィルタ済み

    # 年次集計（ここで先に計算）— 最新年Top5の自動選択に利用
    yearly = _yearly_counts(use, target_mode)
    if yearly.empty:
        st.info("データがありません。")
        return

    # 最新年Top5候補（存在すれば）
    latest_year = int(yearly["発行年"].max()) if not yearly.empty else None
    auto_top: List[str] = []
    if latest_year is not None:
        auto_top = (
            yearly[yearly["発行年"] == latest_year]
            .sort_values("count", ascending=False)[target_mode]
            .head(5).tolist()
        )

    # 最新年Top5を自動選択トグル（中央左）
    with c2:
        # 見た目の整列用スペーサ（Select/MultiSelect の入力ボックス高さに揃える）
        st.markdown('<div style="height:36px;"></div>', unsafe_allow_html=True)
        auto_top5 = st.checkbox(
            "最新年Top5を自動選択",
            value=False,
            key="obj_trend_auto5",
            help="ONにすると、最新年の件数が多い上位5項目を右のボックスに選択状態として入れます。"
        )
        # --- initialize session state for selection to avoid Streamlit default+state collision ---
        if "obj_trend_items" not in st.session_state:
            st.session_state["obj_trend_items"] = []

    # トグルON時は multiselect を最新年Top5でプリセット（セッションに直接セット）
    # すでに同じ年で自動設定済みなら上書きしない（ユーザー編集を尊重）
    if auto_top5 and auto_top:
        if st.session_state.get("_obj_trend_autoset") != latest_year:
            # いまの候補リストに存在するものだけを採用
            # ※ all_items はこの後に計算するため、一旦仮に auto_top を保存し、後段で整合させる
            st.session_state["obj_trend_items"] = auto_top
            st.session_state["_obj_trend_autoset"] = latest_year

    # 候補抽出と順序固定（中央の multiselect で使う）
    all_items_raw = sorted({
        t for v in use.get(target_mode, pd.Series(dtype=str)).fillna("")
        for t in split_multi(v)
    })
    if target_mode == "対象物_top3":
        all_items = _order_options(all_items_raw, TARGET_ORDER)
    else:
        all_items = _order_options(all_items_raw, TYPE_ORDER)

    # セッションに入っている選択肢を、いまの候補に整合（存在しない値を除去）
    if "obj_trend_items" in st.session_state:
        st.session_state["obj_trend_items"] = [x for x in st.session_state["obj_trend_items"] if x in all_items]

    # 表示する項目（中央右）
    with c3:
        sel = st.multiselect(
            "表示する項目（複数可）",
            options=all_items[:1000],
            key="obj_trend_items",
        )

    # 移動平均（右端）
    with c4:
        ma = st.number_input(
            "移動平均（年）",
            min_value=1, max_value=7, value=1, step=1,
            key="obj_trend_ma"
        )

    piv = yearly.pivot_table(index="発行年", columns=target_mode, values="count", aggfunc="sum").fillna(0).sort_index()

    # 現在の選択を適用（トグルON時は上で自動挿入済み）
    if sel:
        piv = piv[[c for c in sel if c in piv.columns]]

    if piv.shape[1] == 0:
        st.info("表示対象がありません。リストから1つ以上選んでください。")
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

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650, fixed_layout: bool = False, node_colors: dict[str,str] | None = None) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("グラフ描画には networkx / pyvis が必要です。")
        return
    if edges.empty:
        st.warning("エッジがありません。")
        return

    import math
    import networkx as nx
    from pyvis.network import Network

    G = nx.Graph()
    for _, r in edges.iterrows():
        s, t, w = str(r["src"]), str(r["dst"]), float(r.get("weight", 1))
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    strength = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}

    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_id = {}
        for i, cset in enumerate(comms):
            for n in cset:
                comm_id[n] = i
    except Exception:
        comm_id = {n: 0 for n in G.nodes()}

    _, palette = _compute_communities_from_edges(edges)
    def node_color(n: str) -> str:
        if node_colors and n in node_colors:
            return node_colors[n]
        cid = int(comm_id.get(n, 0))
        return palette.get(cid, "#999999")

    max_labels = 40
    label_set = set(sorted(G.nodes(), key=lambda n: strength.get(n, 0), reverse=True)[:max_labels])

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    if fixed_layout:
        net.set_options("""
        {"interaction":{"hover":true,"tooltipDelay":200,"zoomView":true,"dragView":true},
         "physics":{"enabled":false},
         "layout":{"improvedLayout":true,"randomSeed":42},
         "nodes":{"shape":"dot"},
         "edges":{"smooth":{"type":"dynamic"}}}
        """)
    else:
        net.set_options("""
        {"interaction":{"hover":true,"tooltipDelay":200,"zoomView":true,"dragView":true},
         "physics":{"stabilization":{"enabled":true,"iterations":200},
                    "barnesHut":{"gravitationalConstant":-25000,"centralGravity":0.2,"springLength":140,"springConstant":0.025,"damping":0.4,"avoidOverlap":0.5}},
         "nodes":{"shape":"dot"},
         "edges":{"smooth":{"type":"dynamic"}}}
        """)

    def size_for(n):
        import math
        s = max(1.0, float(strength.get(n, 1)))
        return max(6.0, min(28.0, 6.0 + 4.0 * math.log1p(s)))

    for n in G.nodes():
        lbl = n if n in label_set else ""
        title = f"{n}<br>総共起重み: {strength.get(n,0):,.0f}"
        net.add_node(
            n,
            label=lbl,
            title=title,
            value=strength.get(n, 0),
            size=size_for(n),
            color=node_color(n),
        )

    def width_for(w):
        return max(1.0, min(10.0, 1.0 + 2.0 * math.log1p(float(w))))
    for s, t, d in G.edges(data=True):
        w = d.get("weight", 1)
        net.add_edge(s, t, value=float(w), width=width_for(w), title=f"共起: {int(w)} 回")

    html = net.generate_html(notebook=False)
    st.components.v1.html(html, height=height_px, scrolling=True)

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
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.0, 1.6, 1.6])
    with c1:
        mode = st.selectbox("ネットワークの種類", ["対象物のみ", "研究タイプのみ", "対象物×研究タイプ"], index=0, key="obj_net_mode")
    with c2:
        topN = st.number_input("表示するノード数（多い順）", min_value=30, max_value=300, value=120, step=10, key="obj_net_topn")
    with c3:
        min_edge = st.number_input("最低共起数（同時出現）", min_value=1, max_value=50, value=3, step=1, key="obj_net_minw")
    # 候補ノード（ネットワーク種類に応じて切替）
    node_options = _node_options_for_mode(df_use, mode)
    with c4:
        include_terms = st.multiselect(
            "必須（選択式）",
            options=node_options,
            default=[],
            key="obj_net_include_sel",
            help="ここで選んだ語を少なくとも1つ含むノードだけを残します。"
        )
    with c5:
        exclude_terms = st.multiselect(
            "除外（選択式）",
            options=node_options,
            default=[],
            key="obj_net_exclude_sel",
            help="ここで選んだ語に該当するノードは除外します。"
        )

    use = df_use
    edges = _build_cooccur_edges(use, mode, int(min_edge))

    # --- 必須／除外（エッジレベルで適用） ---
    if not edges.empty and (include_terms or exclude_terms):
        e = edges.copy()
        if include_terms:
            incl = set(include_terms)
            e = e[(e["src"].isin(incl)) | (e["dst"].isin(incl))]
        if exclude_terms:
            excl = set(exclude_terms)
            e = e[~(e["src"].isin(excl) | e["dst"].isin(excl))]
        edges = e.reset_index(drop=True)

    # --- ノード上限（多い順）を適用 ---
    if not edges.empty and int(topN) > 0:
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    # === 追加: クラスタ（コミュニティ）と代表タイトル ===
    comm_id, palette = _compute_communities_from_edges(edges)
    edge_clusters = []
    edge_colors = []
    ex_titles = []
    for _, r in edges.iterrows():
        a, b = str(r["src"]), str(r["dst"])
        ca, cb = comm_id.get(a, 0), comm_id.get(b, 0)
        c_use = ca if ca == cb else ca
        edge_clusters.append(c_use)
        edge_colors.append(palette.get(c_use, "#999999"))
        ex_titles.append(" / ".join(_example_titles_for_edge(use, mode, a, b, limit=3)))

    edges = edges.copy()
    edges["cluster_id"] = edge_clusters
    edges["cluster_color"] = edge_colors
    edges["example_titles"] = ex_titles

    st.caption(f"エッジ数: {len(edges)}")

    if mode == "対象物のみ":
        col_a, col_b = "対象物A", "対象物B"
    elif mode == "研究タイプのみ":
        col_a, col_b = "研究タイプA", "研究タイプB"
    else:
        col_a, col_b = "対象物", "研究タイプ"

    disp = edges.rename(columns={"src": col_a, "dst": col_b, "weight": "共起回数"}).copy()
    # Build image data URIs for the cluster column
    disp["cluster_img"] = disp["cluster_color"].map(lambda c: _color_square_data_uri(c, 12))

    # Reorder / select columns for display (dataframe)
    disp_view = disp[["cluster_img", col_a, col_b, "共起回数", "example_titles"]].head(200)

    # Render as dataframe with column configs (cluster as ImageColumn)
    try:
        st.dataframe(
            disp_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "cluster_img": st.column_config.ImageColumn("cluster", width="small", help="クラスタ（表・ネットワークと色連動）"),
                col_a: st.column_config.TextColumn(col_a, width="medium"),
                col_b: st.column_config.TextColumn(col_b, width="medium"),
                "共起回数": st.column_config.NumberColumn("共起回数", format="%d", width="small"),
                "example_titles": st.column_config.TextColumn("example_titles", width="large", help="そのペアが同時に登場する論文タイトル（最大3件）"),
            },
        )
    except Exception:
        # Fallback: minimal dataframe without column_config (older Streamlit)
        st.dataframe(disp_view, use_container_width=True, hide_index=True)

    if palette:
        # comm_id は _compute_communities_from_edges で算出済み（ノード -> cluster_id）
        _render_cluster_legend_counts(palette, comm_id)

    # 2) ネットワーク描画
    with st.expander("🕸️ ネットワークを可視化", expanded=False):
        # 可視化専用オプション：レイアウト固定（物理演算を止める）
        fix_layout = st.checkbox(
            "レイアウトを固定",
            value=False,
            key="obj_net_fix_layout",
            help="ONにすると、物理演算を止めてノード位置を固定します。大規模ネットワークでの“ブルブル”を抑え、再描画しても位置が揺れにくくなります。"
        )
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="obj_net_draw"):
                node_colors = {}
                for _, r in edges.iterrows():
                    node_colors[str(r["src"])] = r.get("cluster_color", "#999999")
                    node_colors[str(r["dst"])] = r.get("cluster_color", "#999999")
                _draw_pyvis_from_edges(edges, height_px=680, fixed_layout=fix_layout, node_colors=node_colors)
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