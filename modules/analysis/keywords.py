# modules/analysis/keywords.py
# -*- coding: utf-8 -*-
"""
キーワード分析タブ（完成版・安全な遅延実行＆キャッシュ＋ストップワード対応）

機能（従来どおり）:
① 頻出キーワード分析
   - 年・対象物・研究タイプで絞り込み
   - 出現回数上位をバーチャート表示
   - WordCloud（wordcloud があれば）を任意表示（日本語フォント対応）

② 共起キーワードネットワーク（重いので遅延描画）
   - 同一論文内のキーワード共起を networkx + pyvis で可視化
   - 「ネットワークを描画」ボタン押下時のみ生成
   - ディスクキャッシュ（modules/common/cache_utils.py）対応

③ トレンド分析（経年変化）
   - 年ごとに出現頻度を集計し、TopN語を折れ線で可視化（Plotlyがなければst.line_chart）

注意：
- import時に重い処理を一切走らせません（関数内のみで実行）
- ウィジェットkeyは "kw_*" 接頭で他タブと衝突しないようにしています
"""

from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from pathlib import Path
# --- Robust import for render_filter_bar with error details ---
_HAS_COMMON_FILTERS = False
_FILTER_IMPORT_ERR: str | None = None

# --- Track last filter bar meta result ---
_LAST_FILTER_META: dict[str, Any] = {}
try:
    from modules.common.filters import render_filter_bar  # type: ignore
    _HAS_COMMON_FILTERS = True
except Exception as _e_abs:
    try:
        from ..common.filters import render_filter_bar  # type: ignore
        _HAS_COMMON_FILTERS = True
    except Exception as _e_rel:
        _FILTER_IMPORT_ERR = f"abs[{type(_e_abs).__name__}: {_e_abs}] / rel[{type(_e_rel).__name__}: {_e_rel}]"

def _fallback_filter_bar(df: pd.DataFrame, key_prefix: str = "kw", **kwargs):
    """filters.py が無い/壊れているときの安全フォールバック（UI最小・そのまま返す）"""
    msg = "共通フィルター（filters.py）の読込に失敗したため、データをそのまま表示します。"
    if '_FILTER_IMPORT_ERR' in globals() and _FILTER_IMPORT_ERR:
        msg += f"\n詳細: {_FILTER_IMPORT_ERR}"
    st.warning(msg, icon="⚠️")
    return df

# --- helper: accept tuple/dict return from common filter bar ---
def _df_from_filter_result(res, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """render_filter_bar may return df, (df, ...), or {"df": df}. Be tolerant."""
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

# --- safe wrapper for filter bar ---
def _safe_filter_bar(df: pd.DataFrame,
                     key_prefix: str = "kw",
                     target_order: list[str] | None = None,
                     type_order: list[str] | None = None) -> pd.DataFrame:
    """
    modules.common.filters.render_filter_bar の存在/シグネチャ差/戻り値差を吸収し、常に DataFrame を返す。
    """
    global _LAST_FILTER_META
    if not _HAS_COMMON_FILTERS:
        return _fallback_filter_bar(df, key_prefix=key_prefix)

    # ① 期待シグネチャ
    try:
        res = render_filter_bar(
            df,
            key_prefix=key_prefix,
            target_order=target_order,
            type_order=type_order,
        )
        if isinstance(res, dict):
            _LAST_FILTER_META = res
        else:
            _LAST_FILTER_META = {}
        return _df_from_filter_result(res, df)
    except TypeError:
        # ② 古い/違うシグネチャ（key_prefix のみ）
        try:
            res = render_filter_bar(df, key_prefix=key_prefix)
            if isinstance(res, dict):
                _LAST_FILTER_META = res
            else:
                _LAST_FILTER_META = {}
            return _df_from_filter_result(res, df)
        except TypeError:
            pass
        # ③ 最低限の位置引数のみ
        try:
            res = render_filter_bar(df)
            if isinstance(res, dict):
                _LAST_FILTER_META = res
            else:
                _LAST_FILTER_META = {}
            return _df_from_filter_result(res, df)
        except Exception as e:
            st.warning(f"共通フィルターの呼び出しに失敗しました（{e}）。元データを使用します。", icon="⚠️")
            return df
    except Exception as e:
        st.warning(f"共通フィルターで例外が発生しました（{type(e).__name__}: {e}）。元データを使用します。", icon="⚠️")
        return df
# --- Robustly fetch explicit selections from filter meta or session state ---
def _selected_filters(prefix: str = "kw", df_all: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Fetch explicit selections for 対象物/研究タイプ.
    Priority:
      1) `_LAST_FILTER_META` dict keys (various canonicalizations)
      2) `st.session_state` keys that look like selections (for this prefix)
    Additionally, if the picked list equals "all options" (derived from df_all),
    suppress it by returning an empty list so the banner does not print長 lists.
    """
    def _as_list(x) -> list[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            vals = [str(v).strip() for v in x if str(v).strip()]
        elif isinstance(x, str):
            parts = re.split(r"[,;；、，/／|｜\s\u3000]+", x)
            vals = [p.strip() for p in parts if p.strip()]
        else:
            vals = []
        # remove generic "all" tokens defensively
        ALL_TOKENS = {"全て", "すべて", "すべて選択", "(all)", "all", "ALL"}
        return [v for v in vals if v not in ALL_TOKENS]

    def _dedup_preserve(seq: list[str]) -> list[str]:
        seen = set(); out: list[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    # Collect from meta
    targets: list[str] = []
    types: list[str] = []
    try:
        meta = globals().get("_LAST_FILTER_META", {}) or {}
        cand_tg = [
            "targets", "targets_sel", "targets_selected", "selected_targets",
            "selected_targets_labels", "selected_targets_display", "targets_labels",
            "targets_display", "targets_active",
            f"{prefix}_targets", f"{prefix}_targets_sel", f"{prefix}_targets_selected",
            f"{prefix}_selected_targets", f"{prefix}_selected_targets_labels",
            f"{prefix}_selected_targets_display", f"{prefix}_targets_labels",
            f"{prefix}_targets_display",
            # Japanese label variants (filters using visible label as key)
            "対象物", "対象物_sel", "対象物_selected", "対象物_labels",
            f"{prefix}_対象物", f"{prefix}_対象物_sel", f"{prefix}_対象物_selected", f"{prefix}_対象物_labels",
        ]
        cand_tp = [
            "types", "types_sel", "types_selected", "selected_types",
            "selected_types_labels", "selected_types_display", "types_labels",
            "types_display", "types_active",
            f"{prefix}_types", f"{prefix}_types_sel", f"{prefix}_types_selected",
            f"{prefix}_selected_types", f"{prefix}_selected_types_labels",
            f"{prefix}_selected_types_display", f"{prefix}_types_labels",
            f"{prefix}_types_display",
            # Japanese label variants
            "研究タイプ", "研究タイプ_sel", "研究タイプ_selected", "研究タイプ_labels",
            f"{prefix}_研究タイプ", f"{prefix}_研究タイプ_sel", f"{prefix}_研究タイプ_selected", f"{prefix}_研究タイプ_labels",
        ]
        for k in cand_tg:
            if k in meta and not targets:
                targets = _as_list(meta.get(k))
        for k in cand_tp:
            if k in meta and not types:
                types = _as_list(meta.get(k))
    except Exception:
        pass

    # Fallback: session_state (prefix-aware)
    if not targets or not types:
        try:
            ss = st.session_state
            def _pick(token: str) -> list[str]:
                # try exact keys first
                exact_keys = [
                    f"{prefix}_{token}_sel", f"{token}_sel",
                    f"{prefix}_{token}_labels", f"{token}_labels",
                    f"{prefix}_{token}_display", f"{token}_display",
                    f"{prefix}_{token}", token,
                ]
                # Japanese synonyms for token
                jp_map = {
                    "targets": "対象物",
                    "target": "対象物",
                    "types": "研究タイプ",
                    "type": "研究タイプ",
                }
                if token in jp_map:
                    jp = jp_map[token]
                    exact_keys += [
                        f"{prefix}_{jp}_sel", f"{jp}_sel",
                        f"{prefix}_{jp}_labels", f"{jp}_labels",
                        f"{prefix}_{jp}_display", f"{jp}_display",
                        f"{prefix}_{jp}", jp,
                    ]
                for k in exact_keys:
                    if k in ss:
                        vals = _as_list(ss.get(k))
                        if vals:
                            return vals
                # prefix-scoped scan as a last resort
                for k in ss.keys():
                    if k.startswith(prefix + "_") and (token in k or (token in {"targets","target"} and "対象" in k) or (token in {"types","type"} and "研究" in k)):
                        vals = _as_list(ss.get(k))
                        if vals:
                            return vals
                return []
            if not targets:
                targets = _pick("targets") or _pick("target")
            if not types:
                types = _pick("types") or _pick("type")
        except Exception:
            pass

    # Determine full option universe from df_all (if provided)
    def _all_options_for(col_name: str) -> list[str]:
        if df_all is None or col_name not in df_all.columns:
            return []
        vals = df_all[col_name].fillna("").astype(str).tolist()
        toks: list[str] = []
        for v in vals:
            toks.extend([w.strip() for w in re.split(r"[,;；、，/／|｜\s\u3000]+", v) if w.strip()])
        return _dedup_preserve(toks)

    all_targets = _all_options_for("対象物_top3")
    all_types   = _all_options_for("研究タイプ_top3")

    # Normalize for equality comparison only (case/space-insensitive)
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s).casefold()

    # Suppress only when "exactly all" are selected (not superset/partial)
    if targets and all_targets and set(map(_norm, targets)) == set(map(_norm, all_targets)):
        targets = []
    if types and all_types and set(map(_norm, types)) == set(map(_norm, all_types)):
        types = []

    return _dedup_preserve(targets), _dedup_preserve(types)

def _image_compat(data):
    try:
        # 新しめの Streamlit
        st.image(data, use_container_width=True)
    except TypeError as e:
        # 旧版：use_container_width を知らない
        if "use_container_width" in str(e):
            st.image(data, use_column_width=True)
        else:
            raise

# 並び順（表示順）を固定するための定数
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]


# --- 出典・再現性バナー（分析タブ用：coauthor準拠） ---
def _render_provenance_banner_from_df(
    df_use: pd.DataFrame,
    total_n: int,
    y_from: int | None = None,
    y_to: int | None = None,
    tg_sel: list[str] | None = None,
    tp_sel: list[str] | None = None,
) -> None:
    """
    検索条件の簡潔な要約を1行で表示。
    - 件数：フィルタ後N / 全体
    - 期間：UIで選ばれた年レンジ（無ければdfから推定）
    - 対象物 / 研究タイプ：選択があるときだけ表示（“全部”が選ばれている場合は非表示）
    """
    try:
        n_filtered = len(df_use) if df_use is not None else 0

        # 年レンジ（引数優先。無ければdfから推定）
        if y_from is not None and y_to is not None:
            period = f"{int(y_from)}–{int(y_to)}"
        else:
            years = pd.to_numeric(
                df_use.get("発行年", pd.Series(dtype="object")),
                errors="coerce"
            ).dropna().astype(int) if (df_use is not None and "発行年" in df_use.columns) else pd.Series([], dtype=int)
            period = "—" if years.empty else f"{int(years.min())}–{int(years.max())}"

        # 値の整形（空は非表示）
        def _fmt_list(name: str, vals: list[str] | None, max_items: int = 6):
            if not vals:
                return None
            vs = [str(v).strip() for v in vals if str(v).strip()]
            if not vs:
                return None
            txt = ", ".join(vs[:max_items]) + (" …" if len(vs) > max_items else "")
            return f"{name}：{txt}"

        parts = [f"出典：JBSJ DB（N={n_filtered} / {total_n}）", f"期間：{period}"]
        tg_txt = _fmt_list("対象物", tg_sel)
        tp_txt = _fmt_list("研究タイプ", tp_sel)
        if tg_txt:
            parts.append(tg_txt)
        if tp_txt:
            parts.append(tp_txt)

        st.caption(" ｜ ".join(parts))
    except Exception:
        st.caption(f"出典：JBSJ DB（N={len(df_use) if df_use is not None else 0} / {total_n}）")
# --- extract explicit selections for banner (coauthor準拠の軽量版) ---
def _extract_banner_filters(df_all: pd.DataFrame,
                            key_prefix: str = "kw") -> tuple[int | None, int | None, list[str], list[str]]:
    """
    1) _LAST_FILTER_META に selections があればそれを採用
    2) 無ければ st.session_state の既知キーから拾う
    3) 年レンジは session_state のスライダー値優先、無ければ None
    ※ 「全部選択」かどうかの判定はここでは行わない（空=非表示の方針）
    """
    y_from = y_to = None
    tg_sel: list[str] = []
    tp_sel: list[str] = []

    # 年レンジ（セッション保存のスライダーを最優先）
    try:
        yv = st.session_state.get(f"{key_prefix}_year", None)
        if isinstance(yv, (list, tuple)) and len(yv) == 2:
            y_from, y_to = int(yv[0]), int(yv[1])
    except Exception:
        pass

    # まずはフィルタバーが返したメタを優先
    meta = globals().get("_LAST_FILTER_META", {}) or {}
    def _as_list(x) -> list[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple, set)):
            return [str(v).strip() for v in x if str(v).strip()]
        if isinstance(x, str):
            return [s.strip() for s in re.split(r"[,;；、，/／|｜\s\u3000]+", x) if s.strip()]
        return []

    for k in ["targets", "targets_sel", f"{key_prefix}_targets", f"{key_prefix}_targets_sel", "対象物", f"{key_prefix}_対象物"]:
        if k in meta and not tg_sel:
            tg_sel = _as_list(meta.get(k))
    for k in ["types", "types_sel", f"{key_prefix}_types", f"{key_prefix}_types_sel", "研究タイプ", f"{key_prefix}_研究タイプ"]:
        if k in meta and not tp_sel:
            tp_sel = _as_list(meta.get(k))

    # セッションステートからの補完（メタが空のときのみ）
    if not tg_sel or not tp_sel:
        try:
            ss = st.session_state
            if not tg_sel:
                for k in [f"{key_prefix}_tg", f"{key_prefix}_targets", f"{key_prefix}_対象物", f"{key_prefix}_selected_targets"]:
                    v = ss.get(k)
                    if v:
                        tg_sel = _as_list(v); break
            if not tp_sel:
                for k in [f"{key_prefix}_tp", f"{key_prefix}_types", f"{key_prefix}_研究タイプ", f"{key_prefix}_selected_types"]:
                    v = ss.get(k)
                    if v:
                        tp_sel = _as_list(v); break
        except Exception:
            pass

    return y_from, y_to, tg_sel, tp_sel

# --- 追加: ストップワードとノイズ判定 ---
try:
    from wordcloud import STOPWORDS as WC_STOPWORDS  # type: ignore
    _WC = set(x.casefold() for x in WC_STOPWORDS)
except Exception:
    _WC = set()

STOPWORDS_EN_EXTRA = {
    "and","the","of","to","in","on","for","with","was","were","is","are","be","by","at","from",
    "as","that","this","these","those","an","a","it","its","we","our","you","your","can","may",
    "also","using","use","used","based","between","within","into","than","over","after","before",
    "such","fig","figure","fig.", "table","et","al","etc",
}

STOPWORDS_JA = {
    "こと","もの","ため","など","よう","場合","および","及び","また","これ","それ","この","その",
    "図","表","第","同","一方","または","又は","における","について","に対する"
}

STOPWORDS_ALL = _WC | {s.casefold() for s in STOPWORDS_EN_EXTRA} | STOPWORDS_JA

_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")   # 前後の記号を剥がす
_NUM_RE        = re.compile(r"^\d+(\.\d+)?$")     # 数字のみ
_EN_SHORT_RE   = re.compile(r"^[A-Za-z]{1,2}$")   # 1–2文字の英字（短すぎ）

def _clean_token(tok: str) -> str:
    if tok is None:
        return ""
    t = str(tok).strip()
    if not t:
        return ""
    # 前後の記号を除去
    t = _PUNCT_EDGE_RE.sub("", t)
    if not t:
        return ""
    low = t.casefold()
    if low in {"none", "nan"}:
        return ""
    if _NUM_RE.fullmatch(t):
        return ""
    if _EN_SHORT_RE.fullmatch(t):
        return ""
    if low in STOPWORDS_ALL:
        return ""
    return t

def _get_japanese_font_path() -> str | None:
    """日本語フォントのパスを返す。プロジェクト同梱を最優先。"""
    candidates = [
        "fonts/IPAexGothic.ttf",                            # ← 同梱推奨
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",      # mac
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

 # ==== Optional deps（無くても動く） ====
try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    from wordcloud import WordCloud  # type: ignore
    HAS_WC = True
except Exception:
    HAS_WC = False

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

# Optional: community detection
try:
    from networkx.algorithms.community import greedy_modularity_communities as _greedy_comms  # type: ignore
    HAS_COMMUNITY = True
except Exception:
    HAS_COMMUNITY = False

# 永続キャッシュIO（あれば使う）
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False

# ---- Shared color palette (used for clusters/legend) ----
PALETTE = [
    "#6366F1","#22C55E","#F59E0B","#EF4444","#0EA5E9","#A855F7",
    "#14B8A6","#F97316","#84CC16","#E11D48","#06B6D4","#10B981"
]

# --- Helper: data URI for colored square (PNG or SVG fallback) ---
def _color_square_data_uri(hex_color: str, size: int = 14) -> str:
    """
    Return a data URI of a small colored square (PNG). Falls back to base64 SVG if Pillow is unavailable.
    """
    try:
        from PIL import Image  # type: ignore
        import io, base64
        img = Image.new("RGBA", (size, size), hex_color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        import base64
        svg = (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
            f"<rect width='100%' height='100%' rx='2' fill='{hex_color}'/></svg>"
        )
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"
@st.cache_data(ttl=600, show_spinner=False)
def _compute_node_communities_from_edges(edges: pd.DataFrame) -> dict[str, int]:
    """Return {node: community_id} computed by greedy modularity (if available)."""
    if edges is None or edges.empty or not HAS_NX or not HAS_COMMUNITY:
        return {}
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 1.0)))
    try:
        comms = list(_greedy_comms(G, weight="weight"))
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for gi, nodes in enumerate(comms):
        for n in nodes:
            mapping[str(n)] = gi
    return mapping


# ---- Cached PyVis HTML builder (returns HTML + legend) ----
@st.cache_data(ttl=600, show_spinner=False)
def _build_pyvis_cached(edges: pd.DataFrame, height_px: int = 650, color_mode: str = "community", freeze_layout: bool = False) -> tuple[str, str]:
    """
    Return (html, legend_html) for the given edges using pyvis/networkx.
    Caches the heavy layout/drawing so repeated draws with the same inputs are fast.
    """
    if not (HAS_NX and HAS_PYVIS):
        return ("", "")
    if edges is None or edges.empty:
        return ("", "")

    import math
    import pandas as _pd
    import networkx as _nx  # type: ignore
    from pyvis.network import Network as _Network  # type: ignore

    # ===== グラフ構築（重み付き無向グラフ）=====
    G = _nx.Graph()
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"]); w = float(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    if G.number_of_nodes() == 0:
        return ("", "")

    # ノード指標（可視化用スケーリング）
    deg = dict(G.degree())
    deg_w = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        deg_w[u] += w
        deg_w[v] += w

    # ---- 色分けロジック（既定: community）----
    legend_html: str = ""
    if color_mode == "community" and HAS_COMMUNITY:
        try:
            from networkx.algorithms.community import greedy_modularity_communities as _greedy  # type: ignore
            comms = list(_greedy(G, weight="weight"))
            node_group: dict[str, int] = {}
            for gi, nodes in enumerate(comms):
                for n in nodes:
                    node_group[str(n)] = gi
            chips = []
            for i, nodes in enumerate(comms):
                col = PALETTE[i % len(PALETTE)]
                chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> C{i+1}（{len(nodes)}語）")
            legend_html = " ".join(chips)
        except Exception:
            node_group = {}
    elif color_mode == "degree":
        try:
            dc = _nx.degree_centrality(G)
            vals = _pd.Series(dc).rank(pct=True)
            node_group = {}
            for n, pct in vals.items():
                if pct <= 0.25: g = 0
                elif pct <= 0.5: g = 1
                elif pct <= 0.75: g = 2
                else: g = 3
                node_group[str(n)] = g
            chips = []
            buckets = ["#CBD5E1","#93C5FD","#60A5FA","#2563EB"]
            labels  = ["中心性 下位25%","25–50%","50–75%","上位25%"]
            for col, lab in zip(buckets, labels):
                chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> {lab}")
            legend_html = " ".join(chips)
        except Exception:
            node_group = {}
    else:
        node_group = {}

    # ===== PyVis 描画 =====
    net = _Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-25000, central_gravity=0.15, spring_length=140, spring_strength=0.03, damping=0.18)

    def _scale(value, vmin, vmax, out_min=8, out_max=36):
        if vmax == vmin:
            return (out_min + out_max) / 2
        r = (value - vmin) / (vmax - vmin)
        return out_min + r * (out_max - out_min)

    w_values = list(deg_w.values())
    wmin, wmax = (min(w_values), max(w_values)) if w_values else (0.0, 1.0)

    for n in G.nodes():
        wsum = float(deg_w.get(n, 0.0))
        d = int(deg.get(n, 1))
        size = _scale(wsum if wsum > 0 else d, wmin if wmin > 0 else 0.0, wmax if wmax > 0 else 1.0)
        g = node_group.get(str(n))
        if g is not None:
            gi = int(g)
            G.nodes[n]["group"] = gi
            G.nodes[n]["color"] = PALETTE[gi % len(PALETTE)]
        label = str(n)
        label_short = label if len(label) <= 18 else (label[:16] + "…")
        G.nodes[n]["label"] = label_short
        G.nodes[n]["title"] = f"{label}&lt;br&gt;重み合計: {wsum:.0f} / 度数: {d}"
        G.nodes[n]["value"] = size

    e_w = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    if e_w:
        ew_min, ew_max = (min(e_w), max(e_w))
    else:
        ew_min, ew_max = (1.0, 1.0)
    def _ew_scale(w):
        if ew_max == ew_min:
            return 1.5
        return 1.0 + 4.0 * (w - ew_min) / (ew_max - ew_min)
    for u, v, d in G.edges(data=True):
        d["width"] = _ew_scale(float(d.get("weight", 1.0)))

    net.from_nx(G)
    try:
        net.set_options("""
        {
          "interaction": {
            "hover": true,
            "navigationButtons": false,
            "multiselect": true,
            "tooltipDelay": 120
          },
          "nodes": {
            "shape": "dot",
            "shadow": true,
            "scaling": { "min": 8, "max": 36 },
            "font": { "size": 16, "face": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial" },
            "borderWidth": 1
          },
          "edges": {
            "smooth": {"type": "dynamic"},
            "color": { "opacity": 0.45 }
          },
          "physics": {
            "stabilization": { "enabled": true, "iterations": 220 },
            "barnesHut": { "avoidOverlap": 0.25, "springLength": 140, "springConstant": 0.03, "damping": 0.18 },
            "minVelocity": 0.75
          }
        }
        """)
    except Exception:
        pass

    html = net.generate_html(notebook=False)
    if freeze_layout:
        html = html.replace(
            "network = new vis.Network(container, data, options);",
            "network = new vis.Network(container, data, options);\\nnetwork.once('stabilizationIterationsDone', function () { network.setOptions({ physics: false }); });"
        )
    return (html, legend_html)

@st.cache_data(ttl=600, show_spinner=False)
def _attach_example_titles(df_src: pd.DataFrame, edges: pd.DataFrame, max_titles: int = 3) -> pd.DataFrame:
    """
    Attach 'example_titles' column to edges: up to `max_titles` paper titles in which the pair co-occurred.
    """
    if edges is None or edges.empty:
        return edges if edges is not None else pd.DataFrame(columns=["src","dst","weight","example_titles"])

    # Build per-row unique keyword set & robust title extraction
    rows = []
    for _, r in df_src.iterrows():
        # pick a best-effort title from multiple possible column names
        title = ""
        _title_candidates = [
            "タイトル", "論文タイトル", "論文名", "題名",
            "title", "Title",
            "Japanese Title", "English Title",
            "title_ja", "title_en",
            "タイトル（和）", "タイトル（英）",
            "和文タイトル", "英文タイトル",
        ]
        for _k in _title_candidates:
            if _k in r and pd.notna(r[_k]) and str(r[_k]).strip():
                title = str(r[_k])
                break
        kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
        if not kws:
            continue
        rows.append((set(kws), title))
    out = edges.copy()
    if not rows:
        out["example_titles"] = ""
        return out

    # Ensure the column always exists
    if "example_titles" not in out.columns:
        out["example_titles"] = ""

    examples: list[str] = []
    titles_cache: dict[tuple[str,str], list[str]] = {}

    def _pair_key(a: str, b: str) -> tuple[str,str]:
        return (a, b) if a <= b else (b, a)

    for idx, r in out.iterrows():
        a = str(r["src"]); b = str(r["dst"])
        key = _pair_key(a, b)
        if key in titles_cache:
            cand = titles_cache[key]
        else:
            cand = []
            for kwset, title in rows:
                if a in kwset and b in kwset:
                    if title:
                        cand.append(title)
                if len(cand) >= max_titles:
                    break
            titles_cache[key] = cand
        out.at[idx, "example_titles"] = " / ".join(cand[:max_titles])
    return out


# ========= ユーティリティ =========
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]


def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]


def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    words: List[str] = []
    for c in KEY_COLS:
        if c in row and pd.notna(row[c]):
            for w in split_multi(row[c]):
                cw = _clean_token(w)
                if cw:
                    words.append(cw)
    return words

@st.cache_data(ttl=600, show_spinner=False)
def collect_keywords(df: pd.DataFrame) -> pd.Series:
    """全行からキーワード列を抽出して1本のSeriesに"""
    bags: List[str] = []
    for _, r in df.iterrows():
        bags += _extract_keywords_from_row(r)
    return pd.Series(bags, dtype="object")

@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq(df: pd.DataFrame) -> pd.Series:
    """キーワード頻度（降順）"""
    s = collect_keywords(df)
    if s.empty:
        return pd.Series(dtype=int)
    return s.value_counts().sort_values(ascending=False)

# --- 新規: 頻度をモード別で計算 ---
@st.cache_data(ttl=600, show_spinner=False)
def keyword_freq_by_mode(df: pd.DataFrame, mode: str = "df") -> pd.Series:
    """
    mode: "df" = 登場論文数（1論文に同語が複数回あっても1カウント）
          "tf" = 総出現回数（従来どおり）
    """
    if mode == "df":
        # 1レコード内は重複除去してから集計
        bags: list[str] = []
        for _, r in df.iterrows():
            kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
            bags.extend(kws)
        if not bags:
            return pd.Series(dtype=int)
        return pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False)
    else:
        return keyword_freq(df)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_keyword_counts(df: pd.DataFrame) -> pd.DataFrame:
    """年×語の件数（論文ごと重複除去）"""
    if "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y): 
            continue
        kws = list(dict.fromkeys(_extract_keywords_from_row(r)))
        for k in kws:
            rows.append((int(y), k))
    if not rows:
        return pd.DataFrame(columns=["発行年", "keyword", "count"])
    c = pd.DataFrame(rows, columns=["発行年","keyword"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

# ====== 共起エッジ（重い：キャッシュ対応） ======
@st.cache_data(ttl=600, show_spinner=False)
def build_keyword_cooccur_edges(df: pd.DataFrame, min_edge: int) -> pd.DataFrame:
    """
    同一論文内で共起する語のペアをカウント
    戻り値: ['src','dst','weight']
    """
    rows = []
    for _, r in df.iterrows():
        kws = sorted(set(_extract_keywords_from_row(r)))
        # 全組合せ
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                rows.append((kws[i], kws[j]))
    if not rows:
        return pd.DataFrame(columns=["src","dst","weight"])
    edges = pd.DataFrame(rows, columns=["src","dst"]).value_counts().reset_index(name="weight")
    edges = edges[edges["weight"] >= int(min_edge)].sort_values("weight", ascending=False).reset_index(drop=True)
    return edges

def _freq_to_df(freq: pd.Series, topn: int) -> pd.DataFrame:
    if freq.empty:
        return pd.DataFrame(columns=["キーワード","件数"])
    df = freq.head(int(topn)).reset_index()
    df.columns = ["キーワード","件数"]
    return df

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650, color_mode: str = "community", freeze_layout: bool = False) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("networkx / pyvis が未導入のため、表のみ表示しています。")
        return
    if edges.empty:
        st.warning("対象条件でエッジがありません。")
        return
    html, legend_html = _build_pyvis_cached(edges, height_px=height_px, color_mode=color_mode, freeze_layout=freeze_layout)
    if not html:
        st.info("ネットワークを生成できませんでした。条件を見直してください。")
        return
    st.components.v1.html(html, height=height_px, scrolling=True)
    if legend_html:
        st.markdown("**クラスタ凡例**&nbsp;&nbsp;" + legend_html, unsafe_allow_html=True)
    st.download_button(
        "📥 ネットワークHTMLを保存",
        data=html.encode("utf-8"),
        file_name="keyword_cooccurrence_network.html",
        mime="text/html",
        key="dl_kw_pyvis_html_cached",
        help="単独で開けるHTMLファイルとして保存します（ブラウザでそのまま閲覧可能）。"
    )

# ==== キーワード用：クイックコピー（小さな補助UI。既存UIを崩さない） ====
from typing import List as _ListForCopy

def _render_copy_grid(items: _ListForCopy[str]) -> None:
    """与えられた文字列リストをグリッド表示し、ワンクリックでコピーできる小UI。
    グラフや表の直下に expander として配置する想定。"""
    if not items:
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
    for name in items:
        safe_text = str(name).replace("\\", "\\\\").replace("'", "\\'")
        html += f"""
        <div class=\"copy-chip\">
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
    import streamlit.components.v1 as components  # 局所import（古い環境互換）
    components.html(html, height=200, scrolling=True)

# ==== 汎用: キーワードリストパース&コピーエクスパンダ ====
def _parse_kw_list(s: str) -> list[str]:
    """カンマ/区切り文字/空白で分割してトリム"""
    return [w.strip() for w in re.split(r"[,;；、，/／|｜\s　]+", str(s or "")) if w.strip()]

def _render_copy_expander(items: list[str], title: str) -> None:
    """コピー用エクスパンダ（空なら表示しない）"""
    if not items:
        return
    with st.expander(title, expanded=False):
        _render_copy_grid([str(x) for x in items])

# --- 小プレビュー: 'A, B, C …' 形式で短縮プレビュー ---
def _short_preview(items: list[str], maxn: int = 3) -> str:
    """Return 'A, B, C …' limited preview for captions; empty -> ''."""
    try:
        vals = [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        vals = []
    if not vals:
        return ""
    head = ", ".join(vals[:maxn])
    tail = " …" if len(vals) > maxn else ""
    return head + tail

# ==== 追加：安全表示ヘルパー（UIは変えずに落ちにくく） ====
def safe_show_image(obj: Any) -> None:
    import numpy as np
    import io
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore

    if obj is None:
        st.warning("画像データが None でした。生成に失敗している可能性があります。")
        return

    # Matplotlib Figure -> pyplot
    try:
        import matplotlib.figure
        if isinstance(obj, matplotlib.figure.Figure):
            st.pyplot(obj)  # ここはそのままでOK（旧版でも動く）
            return
    except Exception:
        pass

    # PIL.Image → PNGバイトへ変換してから表示（環境差対策）
    if Image is not None and isinstance(obj, Image.Image):
        try:
            img = obj
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            _image_compat(buf.getvalue())
        except Exception as e:
            st.warning(f"PIL画像の表示で例外が発生しました: {e!s}")
        return

    # NumPy array
    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass
        else:
            st.warning(f"想定外の配列shapeです: {arr.shape}")
            return

        if arr.dtype in (np.float32, np.float64):
            a = arr
            if np.nanmax(a) <= 1.0:
                a = (np.nan_to_num(a) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                a = np.nan_to_num(a).clip(0, 255).astype(np.uint8)
            _image_compat(a)
        elif arr.dtype == np.uint8:
            _image_compat(arr)
        else:
            a = np.nan_to_num(arr).clip(0, 255).astype(np.uint8)
            _image_compat(a)
        return

    # bytes / bytearray
    if isinstance(obj, (bytes, bytearray)):
        _image_compat(obj)
        return

    # 文字列（URL or パス）
    if isinstance(obj, str):
        _image_compat(obj)
        return

    st.warning(f"st.imageが扱えない型でした: {type(obj)}")
    
#
# ========= ① 頻出キーワード =========
def _render_freq_block(df_use: pd.DataFrame) -> None:
    # ---- UI（横並び：表示件数 / 最低総出現回数 / カウント方式）----
    c1, c2, c3 = st.columns([1, 1, 1.6])
    with c1:
        topn = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5, key="kw_freq_topn")
    with c2:
        min_total = st.number_input("最低総出現回数", min_value=1, max_value=100, value=3, step=1, key="kw_freq_min_total")
    with c3:
        count_mode_label = st.radio(
            "カウント方式",
            options=["登場論文数（DF）", "総出現回数（TF）"],
            index=0,
            horizontal=True,
            key="kw_freq_countmode",
            help="DF=1論文に同語が何回出ても1カウント。TF=出現回数そのままカウント。"
        )
        count_mode = "df" if "DF" in count_mode_label else "tf"

    use = df_use

    # 基本頻度（モード別）
    freq = keyword_freq_by_mode(use, mode=count_mode)  # Series: index=keyword, value=count
    if freq.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    # 最低総出現回数（TF/DF いずれのモードでも、合計カウントが閾値未満の語を除外）
    if int(min_total) > 1:
        freq = freq[freq >= int(min_total)]

    freq_df = _freq_to_df(freq, int(topn))
    if freq_df.empty:
        st.info("（フィルタで該当なし）条件を緩めてください。")
        return

    # グラフ
    title_suffix = "（登場論文数）" if count_mode == "df" else "（出現回数）"
    if HAS_PX:
        fig = px.bar(freq_df, x="キーワード", y="件数", text_auto=True, title=f"頻出キーワード{title_suffix}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
        # --- 図下サマリー（頻出：順序・表現修正、グラフ直下に移動） ---
        try:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = _extract_banner_filters(df_all=df_use, key_prefix="kw")
        except Exception:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = None, None, [], []
        _period_txt = f"{int(_y_from)}–{int(_y_to)}" if (_y_from is not None and _y_to is not None) else "—"
        _mode_txt = "DF（登場論文数）" if count_mode == "df" else "TF（総出現回数）"
        _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
        _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
        _parts = [
            f"条件：表示件数：{int(topn)}",
            f"最低回数≧{int(min_total)}",
            _mode_txt,
            f"期間：{_period_txt}",
        ]
        if _tg_preview:
            _parts.append(f"対象物：{_tg_preview}")
        if _tp_preview:
            _parts.append(f"研究タイプ：{_tp_preview}")
        st.caption(" ｜ ".join(_parts))
        # クイックコピー（TopN キーワード）
        _render_copy_expander(freq_df["キーワード"].astype(str).tolist(), "📋 キーワードをすぐコピー")
    else:
        st.bar_chart(freq_df.set_index("キーワード")["件数"])
        # --- 図下サマリー（頻出：順序・表現修正、グラフ直下に移動 / フォールバック） ---
        try:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = _extract_banner_filters(df_all=df_use, key_prefix="kw")
        except Exception:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = None, None, [], []
        _period_txt = f"{int(_y_from)}–{int(_y_to)}" if (_y_from is not None and _y_to is not None) else "—"
        _mode_txt = "DF（登場論文数）" if count_mode == "df" else "TF（総出現回数）"
        _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
        _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
        _parts = [
            f"条件：表示件数：{int(topn)}",
            f"最低回数≧{int(min_total)}",
            _mode_txt,
            f"期間：{_period_txt}",
        ]
        if _tg_preview:
            _parts.append(f"対象物：{_tg_preview}")
        if _tp_preview:
            _parts.append(f"研究タイプ：{_tp_preview}")
        st.caption(" ｜ ".join(_parts))
        _render_copy_expander(freq_df["キーワード"].astype(str).tolist(), "📋 キーワードをすぐコピー")

    # WordCloud（任意）
    with st.expander("☁ WordCloud", expanded=False):
        if HAS_WC:
            if st.button("生成する", key="kw_wc_btn"):
                textfreq = {row["キーワード"]: int(row["件数"]) for _, row in freq_df.iterrows()}
                font_path = _get_japanese_font_path()
                wc = WordCloud(width=900, height=450, background_color="white",
                               collocations=False, prefer_horizontal=1.0,
                               font_path=font_path or None)
                img = wc.generate_from_frequencies(textfreq).to_image()
                safe_show_image(img)
        else:
            st.caption("※ wordcloud が未導入のため非表示です。")

# ========= ② 共起ネットワーク（遅延描画） =========
def _render_cooccur_block(df_use: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.6, 1.6, 0.9])
    with c2:
        min_edge = st.number_input(
            "最低共起数（同時出現）",
            min_value=1, max_value=50, value=3, step=1, key="kw_co_minw",
            help="同じ論文内で2つのキーワードが一緒に登場した回数です。値を上げるほど“よく組み合わせて語られる”強い関係だけが残ります。"
        )
    with c1:
        topN = st.number_input(
            "表示キーワード数",
            min_value=30, max_value=300, value=120, step=10, key="kw_co_topn",
        )
    with c3:
        include_raw = st.text_input(
            "必須キーワード（部分一致・カンマ区切り）",
            value="",
            key="kw_co_include",
            placeholder="例: 酵母, 乳酸菌"
        )
    with c4:
        exclude_raw = st.text_input(
            "除外キーワード（部分一致・カンマ区切り）",
            value="",
            key="kw_co_exclude",
            placeholder="例: 試験, 実験"
        )
    with c5:
        # align checkbox vertically with the text inputs on the left
        st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
        lcc_only = st.checkbox(
            "主要ネットワークのみ",
            value=False,
            key="kw_co_lcc_only",
            help="複数の島に分かれる場合、**一番大きい島だけ**を表示します。"
        )

    # 色分けは自動クラスタを既定で使用（ユーザー選択は廃止）
    _color_mode = "community"

    if not HAS_COMMUNITY:
        st.info("自動クラスタ色分けには networkx の community 機能が必要です。環境で利用できないため、単色表示になります。")

    include_list = [norm_key(x) for x in _parse_kw_list(include_raw)]
    exclude_list = [norm_key(x) for x in _parse_kw_list(exclude_raw)]
    _inc_preview = _short_preview(include_list, maxn=3)
    _exc_preview = _short_preview(exclude_list, maxn=3)

    use = df_use
    # --- キャッシュと描画ロジックはそのまま ---
    edges = build_keyword_cooccur_edges(use, int(min_edge))
    # 1) 必須/除外キーワードフィルタを適用（部分一致）
    if not edges.empty and (include_list or exclude_list):
        def _contains_any(name: str, needles: list[str]) -> bool:
            s = norm_key(name)
            return any(n in s for n in needles)

        if include_list:
            mask_inc = edges["src"].astype(str).map(lambda v: _contains_any(v, include_list)) | \
                       edges["dst"].astype(str).map(lambda v: _contains_any(v, include_list))
            edges = edges[mask_inc]

        if not edges.empty and exclude_list:
            mask_exc = edges["src"].astype(str).map(lambda v: _contains_any(v, exclude_list)) | \
                       edges["dst"].astype(str).map(lambda v: _contains_any(v, exclude_list))
            edges = edges[~mask_exc]

    if not edges.empty and int(topN) > 0:
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    # 2) 最大連結成分のみ（LCC）
    if lcc_only and HAS_NX and not edges.empty:
        try:
            G_tmp = nx.Graph()
            for _, r in edges.iterrows():
                G_tmp.add_edge(str(r["src"]), str(r["dst"]))
            if G_tmp.number_of_nodes() > 0:
                comps = list(nx.connected_components(G_tmp))
                if comps:
                    lcc_nodes = set(max(comps, key=len))
                    edges = edges[edges["src"].astype(str).isin(lcc_nodes) & edges["dst"].astype(str).isin(lcc_nodes)].reset_index(drop=True)
        except Exception as _e:
            st.info(f"LCC 抽出で例外が発生しました（{_e}）。全体を表示します。")
    elif lcc_only and not HAS_NX:
        st.info("LCC 表示は networkx が必要です。networkx が未導入のため全体を表示します。")

    st.caption(f"エッジ数: {len(edges)}")

    # ---- Enrich edges with cluster id & example titles ----
    comm_map = _compute_node_communities_from_edges(edges)
    df_edges = edges.copy()
    if comm_map:
        def _edge_cluster_id(row):
            g1 = comm_map.get(str(row["src"]))
            g2 = comm_map.get(str(row["dst"]))
            if g1 is None and g2 is None:
                return None
            if g1 == g2:
                return g1
            # if nodes belong to different clusters, pick the dominant (src) for display
            return g1 if g1 is not None else g2
        df_edges["cluster_id"] = df_edges.apply(_edge_cluster_id, axis=1)
    else:
        df_edges["cluster_id"] = None
    # 表示用: 色チップ（data URI 画像）— ネットワークと同じ PALETTE
    def _cluster_chip(cid):
        try:
            i = int(cid)
            col = PALETTE[i % len(PALETTE)]
            return _color_square_data_uri(col, size=12)
        except Exception:
            return ""
    df_edges["cluster_img"] = df_edges["cluster_id"].map(_cluster_chip)

    df_edges = _attach_example_titles(df_use, df_edges, max_titles=3)

    # Order columns for display（cluster_img は色付き画像、IDは表には出さない）
    disp_cols = ["cluster_img", "src", "dst", "weight", "example_titles"]
    show_df = df_edges[disp_cols].rename(columns={
        "cluster_img": "cluster",
        "src": "語A",
        "dst": "語B",
        "weight": "共起回数",
        "example_titles": "論文例"
    })
    st.dataframe(
        show_df.head(300),
        use_container_width=True,
        hide_index=True,
        column_config={
            "cluster": st.column_config.ImageColumn(
                "cluster",
                help="自動クラスタリングに基づく色（ネットワークと同期）。",
                width="small"
            ),
            "語A": st.column_config.TextColumn(
                "語A",
                width="small"
            ),
            "語B": st.column_config.TextColumn(
                "語B",
                width="small"
            ),
            "共起回数": st.column_config.NumberColumn(
                "共起回数",
                format="%d",
                width="small"
            ),
            "論文例": st.column_config.TextColumn(
                "論文例",
                help="そのペアが同時に登場した論文タイトルの例（最大3件）",
                width="large"
            ),
        }
    )
    # テーブル用クラスタ凡例（ネットワークと同じ色で同期）
    try:
        present_cids = [int(c) for c in sorted(set(df_edges["cluster_id"].dropna().astype(int)))]
    except Exception:
        present_cids = []
    if present_cids:
        chips = []
        # クラスタ内のノード数（現在のエッジに現れるノードのみ）を数える
        nodes_present = set(df_edges["src"].astype(str)).union(set(df_edges["dst"].astype(str)))
        # comm_map はノード→CID
        counts = {cid: 0 for cid in present_cids}
        for n in nodes_present:
            cid = comm_map.get(str(n))
            if isinstance(cid, int) and cid in counts:
                counts[cid] += 1
        for cid in present_cids:
            col = PALETTE[cid % len(PALETTE)]
            chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> C{cid+1}（{counts.get(cid,0)}語）")
        st.markdown("**クラスタ凡例**&nbsp;&nbsp;" + " ".join(chips), unsafe_allow_html=True)

    # --- 図下サマリー（共起：TopN/閾値/必須・除外/LCC/期間）: 表現・位置修正 ---
    try:
        _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = _extract_banner_filters(df_all=df_use, key_prefix="kw")
    except Exception:
        _y_from, _y_to = None, None
    _period_txt = f"{int(_y_from)}–{int(_y_to)}" if (_y_from is not None and _y_to is not None) else "—"
    _parts = [f"表示キーワード数{int(topN)}", f"最低共起数≧{int(min_edge)}"]
    if _inc_preview:
        _parts.append(f"必須：{_inc_preview}")
    if _exc_preview:
        _parts.append(f"除外：{_exc_preview}")
    if bool(lcc_only):
        _parts.append("主要ネットワークのみ")
    _parts.append(f"期間：{_period_txt}")
    _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
    _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
    if _tg_preview:
        _parts.append(f"対象物：{_tg_preview}")
    if _tp_preview:
        _parts.append(f"研究タイプ：{_tp_preview}")
    st.caption(" ｜ ".join(_parts))

    # クイックコピー（ノード名）
    _nodes = sorted(set(edges["src"].astype(str)).union(set(edges["dst"].astype(str)))) if not edges.empty else []
    _render_copy_expander(_nodes, "📋 ノード名をすぐコピー")

    with st.expander("🕸️ ネットワークを可視化", expanded=False):
        freeze_layout = st.checkbox(
            "レイアウトを固定",
            value=True,
            key="kw_co_freeze",
            help="初期レイアウトが安定したら物理演算を停止します。大規模ネットワークでの“ブルブル”を抑えます。"
        )
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="kw_co_draw"):
                _draw_pyvis_from_edges(edges, height_px=680, color_mode=_color_mode, freeze_layout=freeze_layout)
                # ネットワーク図の直下にも同じサマリーを表示（表現・順序修正）
                _parts = [f"表示キーワード数{int(topN)}", f"最低共起数≧{int(min_edge)}"]
                if _inc_preview:
                    _parts.append(f"必須：{_inc_preview}")
                if _exc_preview:
                    _parts.append(f"除外：{_exc_preview}")
                if bool(lcc_only):
                    _parts.append("主要ネットワークのみ")
                _parts.append(f"期間：{_period_txt}")
                _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
                _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
                if _tg_preview:
                    _parts.append(f"対象物：{_tg_preview}")
                if _tp_preview:
                    _parts.append(f"研究タイプ：{_tp_preview}")
                st.caption(" ｜ ".join(_parts))
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")
# ========= ③ トレンド（経年変化） =========
def _render_trend_block(df_use: pd.DataFrame) -> None:
    # Arrange controls horizontally using st.columns
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.6, 1.6, 1.2])
    with c1:
        topn = st.number_input(
            "表示する語数（TopN）", min_value=5, max_value=50, value=15, step=5, key="kw_trend_topn"
        )
    with c2:
        ma = st.number_input(
            "移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="kw_trend_ma",
            help="年ごとの凸凹をならします。例：3 にすると3年平均。"
        )
    with c3:
        include_raw = st.text_input(
            "必須キーワード（部分一致）",
            value="",
            key="kw_trend_include",
            placeholder="例: 酵母, 乳酸菌"
        )
    with c4:
        exclude_raw = st.text_input(
            "除外キーワード（部分一致）",
            value="",
            key="kw_trend_exclude",
            placeholder="例: 試験, 実験"
        )
    with c5:
        metric = st.radio(
            "指標",
            ["件数", "シェア(%)"],
            index=0,
            horizontal=True,
            key="kw_trend_metric",
            help="シェア(%)は各年の全キーワード合計に対する割合。年ごとの件数差を補正できます。"
        )

    use = df_use
    yearly = yearly_keyword_counts(use)

    # --- 必須/除外キーワードフィルタ ---
    include_list = [norm_key(x) for x in _parse_kw_list(include_raw)]
    exclude_list = [norm_key(x) for x in _parse_kw_list(exclude_raw)]

    if not yearly.empty:
        # 必須キーワードフィルタ
        if include_list:
            mask_inc = yearly["keyword"].astype(str).map(lambda v: any(n in norm_key(v) for n in include_list))
            yearly = yearly[mask_inc]
        # 除外キーワードフィルタ
        if exclude_list:
            mask_exc = yearly["keyword"].astype(str).map(lambda v: any(n in norm_key(v) for n in exclude_list))
            yearly = yearly[~mask_exc]

    if yearly.empty:
        st.info("データがありません。")
        return

    latest_year = yearly["発行年"].max()
    latest_top = (
        yearly[yearly["発行年"] == latest_year]
        .sort_values("count", ascending=False)["keyword"]
        .head(int(topn)).tolist()
    )

    piv = (
        yearly[yearly["keyword"].isin(latest_top)]
        .pivot_table(index="発行年", columns="keyword", values="count", aggfunc="sum")
        .fillna(0).sort_index()
    )

    # シェア(%)に変換（年内合計で割る）
    if metric.startswith("シェア"):
        row_sums = piv.sum(axis=1).replace(0, 1)
        piv = (piv.T / row_sums).T * 100.0

    # 凡例順＝最新年度の多い順で列を並べ替え
    legend_order = [k for k in latest_top if k in piv.columns]
    others = [k for k in piv.columns if k not in legend_order]
    piv = piv[legend_order + sorted(others)]

    if int(ma) > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    if HAS_PX:
        y_label = "シェア(%)" if metric.startswith("シェア") else "件数"
        fig = px.line(
            piv.reset_index().melt(id_vars="発行年", var_name="キーワード", value_name=y_label),
            x="発行年", y=y_label, color="キーワード", markers=True,
            category_orders={"キーワード": legend_order + sorted(others)}
        )
        if metric.startswith("シェア"):
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        # --- 図下サマリー（トレンド：順序・表現修正、グラフ直下に移動） ---
        try:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = _extract_banner_filters(df_all=df_use, key_prefix="kw")
        except Exception:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = None, None, [], []
        _period_txt = f"{int(_y_from)}–{int(_y_to)}" if (_y_from is not None and _y_to is not None) else "—"
        _inc_preview = _short_preview(include_list, maxn=3)
        _exc_preview = _short_preview(exclude_list, maxn=3)
        _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
        _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
        _parts = [
            f"条件：表示する語数：{int(topn)}",
            f"移動平均：{int(ma)}年"
        ]
        if _inc_preview:
            _parts.append(f"必須：{_inc_preview}")
        if _exc_preview:
            _parts.append(f"除外：{_exc_preview}")
        _parts.append(f"指標：{'シェア' if metric.startswith('シェア') else '件数'}")
        _parts.append(f"期間：{_period_txt}")
        if _tg_preview:
            _parts.append(f"対象物：{_tg_preview}")
        if _tp_preview:
            _parts.append(f"研究タイプ：{_tp_preview}")
        st.caption(" ｜ ".join(_parts))
        _legend_items = [c for c in piv.columns if c != "発行年"] if hasattr(piv, 'columns') else []
        _render_copy_expander(_legend_items, "📋 キーワードをすぐコピー")
    else:
        st.line_chart(piv)
        # --- 図下サマリー（トレンド：順序・表現修正、グラフ直下に移動） ---
        try:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = _extract_banner_filters(df_all=df_use, key_prefix="kw")
        except Exception:
            _y_from, _y_to, _tg_sel_tmp, _tp_sel_tmp = None, None, [], []
        _period_txt = f"{int(_y_from)}–{int(_y_to)}" if (_y_from is not None and _y_to is not None) else "—"
        _inc_preview = _short_preview(include_list, maxn=3)
        _exc_preview = _short_preview(exclude_list, maxn=3)
        _tg_preview = _short_preview(_tg_sel_tmp or [], maxn=3)
        _tp_preview = _short_preview(_tp_sel_tmp or [], maxn=3)
        _parts = [
            f"条件：表示する語数：{int(topn)}",
            f"移動平均：{int(ma)}年"
        ]
        if _inc_preview:
            _parts.append(f"必須：{_inc_preview}")
        if _exc_preview:
            _parts.append(f"除外：{_exc_preview}")
        _parts.append(f"指標：{'シェア' if metric.startswith('シェア') else '件数'}")
        _parts.append(f"期間：{_period_txt}")
        if _tg_preview:
            _parts.append(f"対象物：{_tg_preview}")
        if _tp_preview:
            _parts.append(f"研究タイプ：{_tp_preview}")
        st.caption(" ｜ ".join(_parts))
        _legend_items = [c for c in piv.columns if c != "発行年"] if hasattr(piv, 'columns') else []
        _render_copy_expander(_legend_items, "📋 キーワードをすぐコピー")
# ========= エクスポート：タブ本体 =========
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
          .kw-header {
            display: flex;
            align-items: center;      /* 中揃え（縦方向） */
            gap: 12px;
            flex-wrap: wrap;
          }
          .kw-header h2 {
            margin: 0;
          }
          .kw-cap {
            margin: 0;
            font-size: 0.95rem;
            color: #6b7280; /* slate-500 */
            line-height: 1.6;
            white-space: nowrap;
          }
          @media (max-width: 640px) {
            .kw-cap { white-space: normal; }
          }
        </style>
        <div class="kw-header">
          <h2>💬 キーワード</h2>
          <span class="kw-cap">キーワードの頻度・共起・トレンドを確認できます。</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # データ存在チェック
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("データが見つかりません。入力を確認してください。")
        return

    # タブ共通のフィルター（年・対象物・研究タイプ）
    df_use = _safe_filter_bar(
        df,
        key_prefix="kw",
        target_order=TARGET_ORDER,
        type_order=TYPE_ORDER,
    )
    # ▼ 出典バナー（明示選択のみ表示。全部/未選択は非表示）
    _y_from, _y_to, _tg_sel, _tp_sel = _extract_banner_filters(df, key_prefix="kw")
    _render_provenance_banner_from_df(
        df_use,
        total_n=len(df),
        y_from=_y_from,
        y_to=_y_to,
        tg_sel=_tg_sel or None,
        tp_sel=_tp_sel or None,
    )

    tab1, tab2, tab3 = st.tabs([
        "① 頻出キーワード",
        "② 共起ネットワーク",
        "③ トレンド分析",
    ])

    with tab1:
        _render_freq_block(df_use)

    with tab2:
        _render_cooccur_block(df_use)   # ← 遅延描画（ボタン式）

    with tab3:
        _render_trend_block(df_use)