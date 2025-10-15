# modules/common/filters.py
# -*- coding: utf-8 -*-
"""
共通フィルターバー（年 / 対象物 / 研究タイプ）
- 単体 DataFrame を返す（呼び出し側の後方互換重視）
"""


import re
import pandas as pd
import streamlit as st

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

def _order_options(all_options, preferred):
    """
    preferred に含まれるものはその順で先頭に、それ以外は五十音/アルファベット順で後ろに。
    """
    s = set(all_options or [])
    head = [x for x in preferred if x in s]
    tail = sorted([x for x in all_options if x not in preferred])
    return head + tail

# ---- 内部ユーティリティ -------------------------------------------------
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

def _split_multi(s):
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def _norm(s):
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _contains_any(col, needles):
    needles = list(needles or [])
    if not needles:
        return pd.Series([True] * len(col), index=col.index)
    lo = [_norm(n) for n in needles]
    return col.fillna("").astype(str).map(lambda v: any(n in _norm(v) for n in lo))

# ---- メインUI -----------------------------------------------------------
def render_filter_bar(df,
                      key_prefix="flt",
                      target_order=None,
                      type_order=None,
                      show_caption=False,
                      show_reset=False):
    """
    共通フィルターUIを描画し、フィルタ後の DataFrame を返す。
    ※ 引数はダミーを含む（後方互換のため / 未使用でもOK）
    """
    if df is None or df.empty:
        st.info("フィルター対象データがありません。")
        return df

    # 年範囲
    y = pd.to_numeric(df.get("発行年", pd.Series(dtype=float)), errors="coerce")
    if y.notna().any():
        ymin, ymax = int(y.min()), int(y.max())
    else:
        ymin, ymax = 1980, 2025

    # 候補
    targets_all = list({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                        for w in _split_multi(v) if w})
    types_all   = list({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                        for w in _split_multi(v) if w})

    # 並び順を適用（呼び出し側から渡された order があれば優先、なければデフォルト定数）
    _t_order = target_order if target_order else TARGET_ORDER
    _p_order = type_order if type_order else TYPE_ORDER
    targets_all = _order_options(targets_all, _t_order)
    types_all   = _order_options(types_all, _p_order)

    if show_caption:
        st.caption("年・対象物・研究タイプで絞り込みできます。")

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 0.6])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）",
                                 min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax),
                                 key=f"{key_prefix}_year")
    with c2:
        tg = st.multiselect("対象物で絞り込み", options=targets_all, default=[], key=f"{key_prefix}_tg")
    with c3:
        tp = st.multiselect("研究タイプで絞り込み", options=types_all, default=[], key=f"{key_prefix}_tp")
    with c4:
        if show_reset and st.button("リセット", key=f"{key_prefix}_reset"):
            for k in [f"{key_prefix}_year", f"{key_prefix}_tg", f"{key_prefix}_tp"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()
        else:
            st.caption("")

    # 適用
    use = df.copy()
    if "発行年" in use.columns:
        yy = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(yy >= y_from) & (yy <= y_to) | yy.isna()]
    if tg and "対象物_top3" in use.columns:
        use = use[_contains_any(use["対象物_top3"], tg)]
    if tp and "研究タイプ_top3" in use.columns:
        use = use[_contains_any(use["研究タイプ_top3"], tp)]
    return use