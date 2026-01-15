# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
発行年別の論文件数推移（トレンド可視化）
- 共通フィルタ：年レンジ + 移動平均のみ（対象物・研究タイプは共通からは除外）
- サブタブ構成：
  ① 全体推移：全論文の年次推移
  ② 対象物の推移：対象物をチェックボックスで選択、研究タイプでフィルタ可能
  ③ 研究タイプの推移：研究タイプをチェックボックスで選択、対象物でフィルタ可能
- Plotly が無ければ st.line_chart にフォールバック
- 計算はキャッシュ（@st.cache_data）で軽量化
"""

from __future__ import annotations
import re
from typing import List, Tuple

import pandas as pd
import streamlit as st

# 並び順定義
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

def sort_with_order(items, order):
    order_map = {name: i for i, name in enumerate(order)}
    return sorted(items, key=lambda x: order_map.get(x, len(order)))


# ---- Optional deps ----
try:
    import plotly.express as px  # type: ignore
    HAS_PX = True
except Exception:
    HAS_PX = False


# ========= ユーティリティ =========
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

def _apply_year(df: pd.DataFrame, y_from: int, y_to: int) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    return use

@st.cache_data(ttl=600, show_spinner=False)
def _yearly_total_counts(df: pd.DataFrame) -> pd.Series:
    if "発行年" not in df.columns:
        return pd.Series(dtype=int)
    y = pd.to_numeric(df["発行年"], errors="coerce").dropna().astype(int)
    if y.empty:
        return pd.Series(dtype=int)
    return y.value_counts().sort_index()

@st.cache_data(ttl=600, show_spinner=False)
def _yearly_counts_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
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

def _plot_lines_from_pivot(piv: pd.DataFrame, x_label: str = "発行年", legend_order: list[str] | None = None):
    """ピボット(index=年, columns=項目, values=件数)を安全に折れ線描画。
       - x列を必ず用意
       - データ列が1つも無い場合は早期リターン
       - stack→reset_index 後は “列の位置” で [発行年, 項目, 件数] に確実リネーム
       - legend_order が渡されたら、列順・凡例順を固定
    """
    if piv is None or piv.empty:
        st.info("該当データがありません。条件を見直してください。")
        return

    df_plot = piv.copy()

    # ★ 先に列順を固定：存在するものだけを order 順に、余りは末尾へ
    if legend_order:
        head = [c for c in legend_order if c in df_plot.columns]
        tail = [c for c in df_plot.columns if c not in head]
        if head:
            df_plot = df_plot[head + tail]

    # x列（発行年）を必ず作る
    if x_label not in df_plot.columns:
        df_plot.index.name = df_plot.index.name or x_label
        df_plot = df_plot.reset_index()

    # x 以外のデータ列を抽出
    data_cols = [c for c in df_plot.columns if c != x_label]
    if not data_cols:
        st.info("描画対象の項目列がありません。条件を見直してください。")
        return

    # ロング化 → 列名は位置で安全に付け替える
    try:
        df_long = (
            df_plot
            .set_index(x_label)[data_cols]
            .stack(dropna=False)
            .reset_index()
        )
    except Exception as e:
        st.info(f"描画用のデータ整形に失敗しました（整形時例外）。{e}")
        return

    cols = list(df_long.columns)
    if len(cols) < 3:
        st.info("描画用のデータ整形に失敗しました（列の欠落）。条件を見直してください。")
        return
    df_long = df_long.rename(columns={cols[0]: x_label, cols[1]: "項目", cols[2]: "件数"})

    # 型整形
    df_long[x_label] = pd.to_numeric(df_long[x_label], errors="coerce")
    df_long["件数"] = pd.to_numeric(df_long["件数"], errors="coerce")

    # 無効値処理
    df_long = df_long.dropna(subset=[x_label])
    if df_long.empty:
        st.info("該当データがありません。条件を見直してください。")
        return
    if df_long["件数"].notna().sum() == 0:
        df_long["件数"] = 0

    df_long = df_long.fillna({"件数": 0}).sort_values([x_label, "項目"])

    # 可視化
    if HAS_PX:
        if legend_order:
            category_orders = {"項目": [c for c in legend_order if c in df_long["項目"].unique()]}
        else:
            category_orders = None
        fig = px.line(
            df_long, x=x_label, y="件数", color="項目", markers=True,
            category_orders=category_orders
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="項目")
        st.plotly_chart(fig, use_container_width=True)
    else:
        wide = df_long.pivot_table(index=x_label, columns="項目", values="件数", aggfunc="mean").sort_index()
        # ★ st.line_chart でも列順を固定
        if legend_order:
            head = [c for c in legend_order if c in wide.columns]
            tail = [c for c in wide.columns if c not in head]
            wide = wide[head + tail]
        st.line_chart(wide)

def _checkbox_multi(label: str, options: List[str], default_n: int = 10, key_prefix: str = "pub_cb") -> List[str]:
    if not options:
        return []
    st.caption(f"{label}（チェックで選択 / 初期は上位{default_n}件）")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        select_all = st.button("✅ 全選択", key=f"{key_prefix}_all")
    with col_btn2:
        clear_all = st.button("🧹 全解除", key=f"{key_prefix}_clear")
    ncols = 4
    cols = st.columns(ncols)
    selected = []
    for i, opt in enumerate(options):
        col = cols[i % ncols]
        with col:
            init_val = (i < default_n)
            if select_all:
                init_val = True
            if clear_all:
                init_val = False
            checked = st.checkbox(opt, value=init_val, key=f"{key_prefix}_{i}_{opt}")
            if checked:
                selected.append(opt)
    return selected


# ========= メイン描画 =========
def render_temporal_tab(df: pd.DataFrame) -> None:
    st.markdown("## ⏳ 発行年別の論文件数推移")

    if df is None or "発行年" not in df.columns:
        st.info("発行年のデータが見つかりません。")
        return

    ymin, ymax = _year_min_max(df)

    c1, c2 = st.columns([1, 1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax, value=(ymin, ymax), key="pub_year_slider")
    with c2:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="pub_ma")

    use_year = _apply_year(df, y_from, y_to)

    tab1, tab2, tab3 = st.tabs(["① 全体推移", "② 対象物の推移", "③ 研究タイプの推移"])

    # ---- 全体 ----
    with tab1:
        s = _yearly_total_counts(use_year)
        piv = s.to_frame(name="件数")
        piv.index.name = "発行年"
        if int(ma) > 1:
            piv["件数"] = piv["件数"].rolling(window=int(ma), min_periods=1).mean()
        _plot_lines_from_pivot(piv, x_label="発行年")

    # ---- 対象物 ----
    with tab2:
        all_types = {w for v in use_year.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)}
        all_types = sort_with_order(list(all_types), TYPE_ORDER)
        tp_filter = st.multiselect("研究タイプで絞り込み（任意）", all_types, default=[], key="pub_tgt_tp_filter")
        df2 = use_year.copy()
        if tp_filter:
            if "研究タイプ_top3" in df2.columns:
                df2 = df2[col_contains_any(df2["研究タイプ_top3"], tp_filter)]
        all_targets = {w for v in df2.get("対象物_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)}
        all_targets = sort_with_order(list(all_targets), TARGET_ORDER)
        if not all_targets:
            st.info("対象物のデータがありません。")
        else:
            sel_targets = _checkbox_multi("対象物を選択", all_targets, default_n=min(14, len(all_targets)), key_prefix="pub_tgt_cb")
            if not sel_targets:
                st.warning("対象物を1つ以上選んでください。")
            yearly = _yearly_counts_by(df2, "対象物_top3")
            if not yearly.empty:
                yearly = yearly[yearly["対象物_top3"].isin(sel_targets)]
                piv = yearly.pivot_table(index="発行年", columns="対象物_top3", values="count", aggfunc="sum").fillna(0).sort_index()
                if int(ma) > 1:
                    piv = piv.rolling(window=int(ma), min_periods=1).mean()
                # ★ 対象物の順で凡例固定
                legend_order = [x for x in TARGET_ORDER if x in piv.columns] + [c for c in piv.columns if c not in TARGET_ORDER]
                _plot_lines_from_pivot(piv, x_label="発行年", legend_order=legend_order)

    # ---- 研究タイプ ----
    with tab3:
        all_targets = {w for v in use_year.get("対象物_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)}
        all_targets = sort_with_order(list(all_targets), TARGET_ORDER)
        tg_filter = st.multiselect("対象物で絞り込み（任意）", all_targets, default=[], key="pub_typ_tg_filter")
        df3 = use_year.copy()
        if tg_filter:
            if "対象物_top3" in df3.columns:
                df3 = df3[col_contains_any(df3["対象物_top3"], tg_filter)]
        all_types = {w for v in df3.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for w in split_multi(v)}
        all_types = sort_with_order(list(all_types), TYPE_ORDER)
        if not all_types:
            st.info("研究タイプのデータがありません。")
        else:
            sel_types = _checkbox_multi("研究タイプを選択", all_types, default_n=min(11, len(all_types)), key_prefix="pub_typ_cb")
            if not sel_types:
                st.warning("研究タイプを1つ以上選んでください。")
            yearly = _yearly_counts_by(df3, "研究タイプ_top3")
            if not yearly.empty:
                yearly = yearly[yearly["研究タイプ_top3"].isin(sel_types)]
                piv = yearly.pivot_table(index="発行年", columns="研究タイプ_top3", values="count", aggfunc="sum").fillna(0).sort_index()
                if int(ma) > 1:
                    piv = piv.rolling(window=int(ma), min_periods=1).mean()
                # ★ 研究タイプの順で凡例固定
                legend_order = [x for x in TYPE_ORDER if x in piv.columns] + [c for c in piv.columns if c not in TYPE_ORDER]
                _plot_lines_from_pivot(piv, x_label="発行年", legend_order=legend_order)