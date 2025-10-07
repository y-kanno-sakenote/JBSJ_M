# modules/analysis/temporal.py
# -*- coding: utf-8 -*-
"""
⏳ 経年変化（中心性スコアの移動窓トレンド）

このタブは、特定期間の共著ネットワークから「研究者の中心性」が
時間とともにどう移り変わったかを、移動窓（スライドする年区間）で可視化します。

■ できること
- フィルタ：対象物・研究タイプで論文集合を絞り込み（部分一致）
- 指標選択：degree / betweenness / eigenvector を切り替え
- 期間設定：ウィンドウ幅（年）・シフト幅（年）・開始年で移動窓を定義
- 可視化：各移動窓で算出した中心性スコアの推移を折れ線表示
- 実務補助：直近ウィンドウのランキング（著者 / 共著数 / 中心性スコア）を確認

■ 用語ざっくり
- 次数中心性（degree）：どれだけ多くの相手とつながっているか（横の広さ）
- 媒介中心性（betweenness）：研究者同士の橋渡しの度合い（ネットワークの要）
- 固有ベクトル中心性（eigenvector）：影響力のある相手とつながっているほど高い（影響の質）
  ※ networkx が未導入の場合は、近似として「共著数の合計」をスコアに使います。

■ 表示の読み方
- 折れ線1本＝1人の研究者。ラインが上昇すればその期間で影響力が増加。
- フィルタを使うと、特定領域（例：清酒×微生物）だけの“リーダー交代”が見えます。
"""

from __future__ import annotations
import itertools
import re
import pandas as pd
import streamlit as st

# ---- Optional deps（無ければ自動フォールバック）----
try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ========= 文字列ユーティリティ =========
def _split_authors(cell) -> list[str]:
    """著者セルを区切り記号で分割。空要素は除去。"""
    if cell is None:
        return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜]+", str(cell)) if w.strip()]

def _split_multi(s):
    """'清酒; ワイン / ビール' のような複合文字列を分割。"""
    if not s:
        return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

def _norm_key(s: str) -> str:
    """小文字化＋全角/連続空白の正規化（部分一致用）。"""
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _col_contains_any(df_col: pd.Series, needles: list[str]) -> pd.Series:
    """列に対して needles のいずれかが部分一致するか（正規化して評価）。"""
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [_norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = _norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)


# ========= 年レンジユーティリティ =========
def _year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """DFから発行年の最小/最大を取得。無い場合はデフォルト値。"""
    if "発行年" in df.columns:
        y = pd.to_numeric(df["発行年"], errors="coerce")
        if y.notna().any():
            return int(y.min()), int(y.max())
    return 1980, 2025


# ========= エッジ構築（移動窓で使い回すのでキャッシュ） =========
@st.cache_data(ttl=600, show_spinner=False)
def _build_edges(df: pd.DataFrame, y_from: int, y_to: int) -> pd.DataFrame:
    """
    [y_from, y_to] の範囲で共著エッジを構築。
    返り値: ['src','dst','weight']
    """
    use = df.copy()

    # 年レンジで絞り込み（欠損年は通す＝レビュー等を残す）
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]

    # 著者ペアを重み付きでカウント
    rows = []
    for authors in use.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(_split_authors(authors)))
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


# ========= 中心性スコア（networkx 無しでも動く） =========
def _centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.Series:
    """
    エッジ→中心性スコア（Series: index=author, value=score）
    - networkx 無し：重み付き次数（共著重みの合計）で近似
    """
    if edges.empty:
        return pd.Series(dtype=float)

    if not HAS_NX:
        deg = (
            pd.concat(
                [edges.groupby("src")["weight"].sum(), edges.groupby("dst")["weight"].sum()],
                axis=1,
            )
            .fillna(0)
            .sum(axis=1)
            .sort_values(ascending=False)
        )
        deg.name = "score"
        return deg

    # networkx あり：本格計算
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
            cen = nx.degree_centrality(G)  # フォールバック
    else:
        cen = nx.degree_centrality(G)

    s = pd.Series(cen, dtype=float).sort_values(ascending=False)
    s.name = "score"
    return s


# ========= 時系列（移動窓）スコア =========
@st.cache_data(ttl=600, show_spinner=False)
def _sliding_window_scores(
    df: pd.DataFrame,
    metric: str,
    start_year: int,
    win: int,
    step: int,
    ymax: int,
) -> pd.DataFrame:
    """
    start_year から win 年の窓を step 年ずつ右へスライドしながら中心性を算出。
    返り値: long形式 ['window','author','score']（window は "YYYY-YYYY" 文字列）
    """
    records = []
    s = start_year
    while s <= ymax - win + 1:
        e = s + win - 1
        edges = _build_edges(df, s, e)
        scores = _centrality_from_edges(edges, metric=metric)
        if not scores.empty:
            rec = pd.DataFrame(
                {"window": f"{s}-{e}", "author": scores.index, "score": scores.values}
            )
            records.append(rec)
        s += step

    if not records:
        return pd.DataFrame(columns=["window", "author", "score"])
    return pd.concat(records, ignore_index=True)


# ========= メイン描画 =========
def render_temporal_tab(df: pd.DataFrame, use_disk_cache: bool = True) -> None:
    """
    UIの流れ：
      1) フィルタ（対象物 / 研究タイプ）
      2) 期間と指標の設定（ウィンドウ幅・シフト幅・開始年・上位著者数）
      3) 時系列スコアを算出 → 折れ線で推移を表示
      4) 直近ウィンドウのランキングも併記（実務での確認用）
    """
    st.markdown("## ⏳ 研究ネットワークの経年変化（移動窓）")

    if df is None or "著者" not in df.columns:
        st.info("著者データが見つかりません。")
        return

    # ---- 年範囲の自動推定（安全デフォルトへフォールバック） ----
    ymin, ymax = _year_bounds(df)

    # ---- 1) 対象物/研究タイプ で軽量フィルタ（部分一致）----
    st.markdown("### 🔍 絞り込み条件（任意）")
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        tg_raw = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in _split_multi(v)}
        targets_sel = st.multiselect("対象物", sorted(tg_raw), default=[], key="temporal_tg")
    with c_f2:
        tp_raw = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in _split_multi(v)}
        types_sel = st.multiselect("研究タイプ", sorted(tp_raw), default=[], key="temporal_tp")

    df_filt = df.copy()
    if targets_sel and "対象物_top3" in df_filt.columns:
        df_filt = df_filt[_col_contains_any(df_filt["対象物_top3"], targets_sel)]
    if types_sel and "研究タイプ_top3" in df_filt.columns:
        df_filt = df_filt[_col_contains_any(df_filt["研究タイプ_top3"], types_sel)]

    if df_filt.empty:
        st.warning("条件に一致するデータがありません。フィルタを調整してください。")
        return

    # ---- 2) ウィンドウ設定 + 指標選択 ----
    st.markdown("### ⚙️ 期間と指標の設定")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        metric = st.selectbox(
            "中心性指標",
            ["degree", "betweenness", "eigenvector"],
            index=0,
            key="temporal_metric",
            help="networkx 未導入時は“共著数の合計”で代替します。",
            format_func=lambda x: {"degree": "次数", "betweenness": "媒介", "eigenvector": "固有ベクトル"}[x],
        )
    with c2:
        win = st.number_input(
            "ウィンドウ幅（年）",
            min_value=2,
            max_value=max(2, ymax - ymin + 1),
            value=min(5, max(2, ymax - ymin + 1)),
            step=1,
            key="temporal_win",
            help="1つの窓の年数。例：5年なら“2000–2004”で1区間。",
        )
    with c3:
        step = st.number_input(
            "シフト幅（年）",
            min_value=1,
            max_value=max(1, ymax - ymin + 1),
            value=1,
            step=1,
            key="temporal_step",
            help="窓をどれだけ右へ進めるか。1なら2000–2004 → 2001–2005 → …",
        )
    max_start = max(ymin, ymax - int(win) + 1)
    with c4:
        start_year = st.slider(
            "開始年",
            min_value=ymin,
            max_value=max_start,
            value=min(ymin, max_start),
            step=1,
            key="temporal_start",
            help="最初の窓の左端。ここからシフト幅ずつ右へ評価します。",
        )
    with c5:
        top_k = st.number_input(
            "上位著者数",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            key="temporal_topk",
            help="可視化対象の著者数（ウィンドウ全体で重要度の高い順）。",
        )

    end_year = start_year + int(win) - 1
    st.caption(f"📅 対象期間: **{start_year}–{end_year}**（{win}年・シフト {step}年）")

    # ---- 3) スコア計算（キャッシュ有り）----
    with st.spinner("時系列スコアを計算中..."):
        # ※ 移動窓の内部は _build_edges / _centrality_from_edges でキャッシュ済み
        scores_long = _sliding_window_scores(
            df=df_filt,
            metric=metric,
            start_year=start_year,
            win=int(win),
            step=int(step),
            ymax=ymax,
        )

    if scores_long.empty:
        st.info("該当期間で共著ネットワークが構築できませんでした。年範囲やフィルタを調整してください。")
        return

    # 可視化対象の著者を上位に絞る（全期間の最大スコアでソート）
    top_authors = (
        scores_long.groupby("author")["score"]
        .max()
        .sort_values(ascending=False)
        .head(int(top_k))
        .index
        .tolist()
    )
    plot_df = scores_long[scores_long["author"].isin(top_authors)].copy()

    # ---- 4) 折れ線可視化 ----
    st.markdown("### 📈 中心性スコアの推移（移動窓）")
    if HAS_PLOTLY:
        fig = px.line(
            plot_df, x="window", y="score", color="author", markers=True, template="plotly_white",
            labels={"window": "ウィンドウ（年区間）", "score": "中心性スコア", "author": "著者"},
        )
        fig.update_layout(legend_title_text="著者", height=460, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = plot_df.pivot(index="window", columns="author", values="score").fillna(0.0)
        st.line_chart(pivot)

    with st.expander("📄 データを表示", expanded=False):
        st.dataframe(plot_df.sort_values(["window", "score"], ascending=[True, False]), hide_index=True)

    # ---- 5) 直近ウィンドウのランキング（現況確認）----
    st.markdown("### 🔝 直近ウィンドウの上位")
    last_from = max(end_year - int(win) + 1, start_year)
    last_to = end_year
    edges_last = _build_edges(df_filt, last_from, last_to)
    # 直近は表を見やすく（共著数も併記）
    # 共著数＝重み合計（networkx無でも計算可）
    deg_last = (
        pd.concat(
            [edges_last.groupby("src")["weight"].sum(), edges_last.groupby("dst")["weight"].sum()],
            axis=1,
        )
        .fillna(0)
        .sum(axis=1)
        .rename("coauth_count")
        .reset_index()
        .rename(columns={"index": "author"})
    )
    scores_last = _centrality_from_edges(edges_last, metric=metric).rename("score").reset_index().rename(columns={"index": "author"})
    rank_last = pd.merge(scores_last, deg_last, on="author", how="left").fillna({"coauth_count": 0})
    rank_last = rank_last.sort_values("score", ascending=False).head(30)
    rank_last = rank_last.rename(columns={"author": "著者", "score": "中心性スコア", "coauth_count": "共著数"})
    st.dataframe(rank_last, use_container_width=True, hide_index=True)

    st.caption("※ 指標の意味：次数=つながりの数 / 媒介=橋渡し度 / 固有ベクトル=影響力（影響力の高い相手との結び付き）")