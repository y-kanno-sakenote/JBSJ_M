# modules/analysis/coauthor/ui.py
from __future__ import annotations
import pandas as pd
import streamlit as st


from modules.common.state import GlobalFilters
from modules.common import banners, copyui
from .filters_adapter import adapt_filter_bar, augment_with_session_state, split_authors
from .compute import (author_total_counts, yearly_author_counts,
                      build_coauthor_edges, centrality_from_edges)
from .network_view import draw_network

# --- Optional deps / utilities for network summary & caching ---
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv  # type: ignore
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False

# Optional: Japanese reading (よみ) for author labels
try:
    from pykakasi import kakasi  # type: ignore
    _KKS = kakasi(); _KKS.setMode('J','H'); _KKS.setMode('K','H'); _KKS.setMode('H','H')
    HAS_KAKASI = True
except Exception:
    HAS_KAKASI = False
    _KKS = None  # type: ignore

# Fallback: optionally load precomputed readings from data/authors_readings.csv
_AUTHOR_READINGS: dict | None = None

def _ensure_author_readings() -> None:
    """Lazy-load a CSV file with author reading mappings (name -> reading).

    Tries a couple of likely locations (project ./data/ and package-relative). No exception
    is raised; failures silently leave the mapping empty.
    """
    global _AUTHOR_READINGS
    if _AUTHOR_READINGS is not None:
        return
    _AUTHOR_READINGS = {}
    try:
        from pathlib import Path
        import pandas as _pd
        cand = [Path.cwd() / "data" / "authors_readings.csv", Path(__file__).resolve().parents[3] / "data" / "authors_readings.csv"]
        for p in cand:
            if p.exists():
                try:
                    df = _pd.read_csv(p, encoding="utf-8")
                    cols = [c for c in df.columns]
                    if len(cols) >= 2:
                        key_col, val_col = cols[0], cols[1]
                        for _, r in df.iterrows():
                            name = str(r.get(key_col, "")).strip()
                            yomi = str(r.get(val_col, "")).strip()
                            if name:
                                _AUTHOR_READINGS[name] = yomi
                    break
                except Exception:
                    # ignore and try next candidate
                    pass
    except Exception:
        _AUTHOR_READINGS = {}

# Shared palette (sync with network colors if needed)
_PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#8c6d31"
]

def _author_label(name: str) -> str:
    """漢字｜よみ（pykakasiがあれば）"""
    if HAS_KAKASI and _KKS is not None:
        try:
            yomi = _KKS.getConverter().do(str(name))
            if yomi:
                return f"{name}｜{yomi}"
        except Exception:
            pass
    # fallback: precomputed readings file (デプロイ環境で pykakasi 未導入の場合の補助)
    try:
        _ensure_author_readings()
        if _AUTHOR_READINGS and name in _AUTHOR_READINGS and _AUTHOR_READINGS[name]:
            return f"{name}｜{_AUTHOR_READINGS[name]}"
    except Exception:
        pass
    return str(name)

def _color_square_data_uri(hex_color: str, size: int = 16) -> str:
    """Small colored square as SVG data URI (no Pillow dependency)."""
    import base64
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}'>"
        f"<rect width='{size}' height='{size}' fill='{hex_color}' rx='3' ry='3'/></svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"

_HEADER_HTML = """
<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin: 0 0 4px 0;">
  <h2 style="margin:0; line-height:1; font-weight:600;">👨‍🔬 研究者</h2>
  <div style="margin:0 0 2px 0; line-height:1.2; opacity:0.8; font-size:0.95rem;">
    著者別の論文数・共著ネットワーク・トレンドを確認できます。
  </div>
</div>
"""

_METRIC_JA = {
    "degree": "次数（つながりの数）",
    "betweenness": "媒介（橋渡し度）",
    "eigenvector": "固有ベクトル（影響力）",
}

def _summarize(y_from: int, y_to: int, tg_sel, tp_sel) -> str:
    gf = GlobalFilters(y_from, y_to, tg_sel, tp_sel)
    return banners.summarize(gf)

def render_coauthor_tab(df: pd.DataFrame, use_disk_cache: bool = False):
    st.markdown(_HEADER_HTML, unsafe_allow_html=True)
    if df is None or ("著者" not in df.columns):
        st.warning("著者データが見つかりません。")
        return

    df_use, y_from, y_to, tg_sel, tp_sel = adapt_filter_bar(df)
    y_from, y_to, tg_sel, tp_sel = augment_with_session_state(y_from, y_to, tg_sel, tp_sel, key_prefix="authors")
    banners.render_provenance(df_use, len(df), GlobalFilters(y_from, y_to, tg_sel, tp_sel))

    tab_count, tab_network, tab_trend = st.tabs(["① 論文数", "② 共著ネットワーク", "③ トレンド分析"])

    # ===== ① 論文数 =====
    with tab_count:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            mode = st.radio("著者数フィルタ", ["すべて", "単著のみ", "共著のみ"], horizontal=True, key="res_cnt_mode")
        with c2:
            period = st.radio("集計期間", ["累計", "直近1年", "直近3年", "直近5年"], horizontal=True, key="res_cnt_period")
        with c3:
            position = st.multiselect("著者ポジション", ["筆頭のみ","責任著者のみ"], key="res_cnt_position")
        with c4:
            top_n = st.number_input("ランキング件数", min_value=5, max_value=200, value=50, step=5, key="res_cnt_topn")

        df_rank = df_use
        if period != "累計" and "発行年" in df_rank.columns:
            years = pd.to_numeric(df_rank["発行年"], errors="coerce")
            span = {"直近1年":1, "直近3年":3, "直近5年":5}[period]
            y_max = int(years.max()) if years.notna().any() else None
            if y_max is not None:
                df_rank = df_rank[(years >= y_max - span + 1) & (years <= y_max)]

        if mode != "すべて":
            df_rank = df_rank.copy()
            df_rank["著者数"] = df_rank["著者"].fillna("").map(lambda s: len(split_authors(s)))
            if mode == "単著のみ":
                df_rank = df_rank[df_rank["著者数"] == 1]
            else:
                df_rank = df_rank[df_rank["著者数"] >= 2]

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
                if chosen:
                    bags.extend(list(dict.fromkeys(chosen)))
            s = pd.Series(bags, dtype="object").value_counts().sort_values(ascending=False) if bags else pd.Series(dtype=int)
        else:
            s = author_total_counts(df_rank)

        if s.empty:
            st.info("条件に合うデータがありません。")
        else:
            rank = s.reset_index()
            rank.columns = ["著者", "論文数"]
            rank = rank.sort_values(["論文数", "著者"], ascending=[False, True])
            rank_shown = rank.head(int(top_n))

            left, right = st.columns([1.0, 1.1])
            with left:
                st.dataframe(rank_shown[["著者", "論文数"]], use_container_width=True, hide_index=True, height=420)
            with right:
                try:
                    import plotly.express as px
                    bar_df = rank.head(10).sort_values("論文数", ascending=False)
                    fig = px.bar(bar_df, x="論文数", y="著者", orientation="h", text_auto=True, title="著者Top10")
                    fig.update_layout(margin=dict(l=6, r=6, t=40, b=6), height=420, xaxis_title=None, yaxis_title=None)
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.bar_chart(rank.set_index("著者")["論文数"].head(10))

            parts = []
            if mode != "すべて": parts.append(mode)
            if position: parts.append("・".join(position))
            parts.append(period)
            st.caption(f"条件：{'・'.join(parts)} ｜ ランキング件数：{int(top_n)} ｜ " + _summarize(y_from, y_to, tg_sel, tp_sel))


    # ===== ② 共著ネットワーク =====
    with tab_network:
        # オートコンプリート用：著者候補（頻度上位から最大600件）
        try:
            _auth_freq = author_total_counts(df_use)
            _author_names = _auth_freq.index.tolist()[:600]
        except Exception:
            _bags = []
            for a in df_use.get("著者", pd.Series(dtype=str)).fillna(""):
                _bags.extend(split_authors(a))
            _author_names = sorted(list(dict.fromkeys(_bags)))[:600]

        # 表示は「漢字｜よみ」。検索は読みでも可（Streamlitは表示文字列で検索）
        _author_labels = [_author_label(n) for n in _author_names]
        _label_to_name = {lbl: nm for lbl, nm in zip(_author_labels, _author_names)}

        # メトリック・ランキング件数・最小共著回数・必須・除外
        c4, c5, c6, c7, c8 = st.columns([1,1,1,2,2])
        with c4:
            metric = st.selectbox(
                "中心性指標",
                ["degree", "betweenness", "eigenvector"],
                index=0,
                format_func=lambda x: _METRIC_JA.get(x, x),
                help="networkx が未導入の場合は簡易スコア（共著数の合計）で代替します。",
                key="res_net_metric",
            )
        with c5:
            top_n = st.number_input(
                "ランキング件数",
                min_value=5, max_value=100, value=30, step=5,
                key="res_net_topn",
            )
        with c6:
            min_w = st.number_input(
                "最小共著回数（w ≥）",
                min_value=1, max_value=20, value=2, step=1,
                key="res_net_minw",
                help="この回数未満の共著エッジは非表示。値を上げるほど“よく組む”強い関係だけが残ります。"
            )
        with c7:
            must_sel_labels = st.multiselect(
                "必須（著者名・読みで検索可）",
                options=_author_labels,
                default=[],
                key="res_net_must_ms",
            )
            must_sel = [_label_to_name.get(x, x) for x in must_sel_labels]
        with c8:
            excl_sel_labels = st.multiselect(
                "除外（著者名・読みで検索可）",
                options=_author_labels,
                default=[],
                key="res_net_excl_ms",
            )
            excl_sel = [_label_to_name.get(x, x) for x in excl_sel_labels]

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
            # 新版 compute API は年/対象物/タイプを渡す（後方互換があれば try/except）
            try:
                edges = build_coauthor_edges(df_use, y_from, y_to, tg_sel, tp_sel)
            except TypeError:
                edges = build_coauthor_edges(df_use)
            if use_disk_cache and HAS_DISK_CACHE and edges is not None:
                save_csv(edges, cache_csv_path("coauthor_edges", cache_key))

        # --- 必須／除外（サジェスト選択）フィルタをエッジに適用 ---
        if edges is not None and not edges.empty:
            if must_sel:
                ms = set(must_sel)
                edges = edges[edges.apply(lambda r: (r["src"] in ms) or (r["dst"] in ms), axis=1)]
            if excl_sel:
                es = set(excl_sel)
                edges = edges[~edges.apply(lambda r: (r["src"] in es) or (r["dst"] in es), axis=1)]
            edges = edges.reset_index(drop=True)

        if edges is None or edges.empty:
            st.info("共著関係が見つかりませんでした。条件を調整してください。")
        else:
            # --- ノード色マップ（ネットワーク描画と同期用） ---
            node_color_map = None
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:6px; margin:6px 0 2px 0;">
                  <span style="font-weight:600; font-size:0.95rem; opacity:0.9;">🔝 研究者ネットワーク要約（クラスタ色連動）</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            rank = centrality_from_edges(edges, metric=metric).head(int(top_n))
            st.caption("※ 指標の意味：次数＝つながりの数｜媒介＝橋渡し度｜固有ベクトル＝影響力（有力者との結び付き）")
            st.caption("※ 必須：選んだ著者を含むエッジだけ表示／除外：含むエッジを除外（漢字・“よみ”で検索可）。")

            # --- 中心著者＋近傍サマリー（クラスタ色と連動） ---
            try:
                if HAS_NX:
                    _edges_for_summary = edges[edges["weight"] >= int(min_w)].copy()
                    Gsum = nx.Graph()
                    for _, r in _edges_for_summary.iterrows():
                        Gsum.add_edge(str(r["src"]), str(r["dst"]), weight=float(r["weight"]))
                    if Gsum.number_of_nodes() > 0:
                        try:
                            from networkx.algorithms.community import louvain_communities
                            _comms = list(louvain_communities(Gsum, weight="weight", resolution=1.3))
                        except Exception:
                            from networkx.algorithms.community import greedy_modularity_communities
                            _comms = list(greedy_modularity_communities(Gsum, weight="weight"))
                        _comm_id = {}
                        for i, cset in enumerate(_comms):
                            for n in cset:
                                _comm_id[n] = i

                        _central_nodes = rank["著者"].tolist()
                        _rows = []
                        for author in _central_nodes:
                            if author not in Gsum:
                                continue
                            partners = []
                            for nbr in Gsum.neighbors(author):
                                w = float(Gsum[author][nbr].get("weight", 1.0))
                                partners.append((nbr, w))
                            partners.sort(key=lambda x: (-x[1], x[0]))
                            uniq_partners = [p for p, _ in partners]
                            top_partners = uniq_partners[:3]

                            titles = []
                            _title_cols = ["タイトル", "論文タイトル", "title", "Title", "題名"]
                            for _, rr in df_use.iterrows():
                                names = list(dict.fromkeys(split_authors(rr.get("著者", ""))))
                                if (author in names) and any(tp in names for tp in top_partners):
                                    t = ""
                                    for _c in _title_cols:
                                        if _c in rr and pd.notna(rr[_c]) and str(rr[_c]).strip():
                                            t = str(rr[_c]).strip()
                                            break
                                    if t:
                                        titles.append(t)
                                if len(titles) >= 3:
                                    break

                            cid = int(_comm_id.get(author, 0))
                            ccolor = _PALETTE[cid % len(_PALETTE)]

                            _rows.append({
                                "cluster_id": cid,
                                "cluster_color": ccolor,
                                "cluster": " ",
                                "中心著者": author,
                                "相手著者数": len(set(uniq_partners)),
                                "代表相手（上位3名）": "／".join(top_partners),
                                "example_titles": "／".join(titles)
                            })

                        if _rows:
                            _sum_df = pd.DataFrame(
                                _rows,
                                columns=[
                                    "cluster",
                                    "中心著者",
                                    "相手著者数",
                                    "代表相手（上位3名）",
                                    "example_titles",
                                    "cluster_id",
                                    "cluster_color",
                                ],
                            )
                            _rank_for_merge = rank[["著者", "共著数"]].rename(columns={"著者": "中心著者"})
                            _merged = pd.merge(_sum_df, _rank_for_merge, on="中心著者", how="left")

                            _disp = _merged.rename(columns={"中心著者": "著者", "example_titles": "論文例"}).copy()
                            node_color_map = {str(a): str(c) for a, c in _merged[["中心著者", "cluster_color"]].dropna().values}
                            _disp = _disp[["cluster", "著者", "共著数", "相手著者数", "代表相手（上位3名）", "論文例"]]
                            _disp["共著数"] = _disp["共著数"].fillna(0).astype(int)
                            _disp["相手著者数"] = _disp["相手著者数"].fillna(0).astype(int)

                            _disp = _merged.rename(columns={"中心著者": "著者", "example_titles": "論文例"}).copy()
                            _disp["cluster_img"] = _merged["cluster_color"].map(lambda c: _color_square_data_uri(c))
                            _disp = _disp[["cluster_img", "著者", "共著数", "相手著者数", "代表相手（上位3名）", "論文例"]]
                            _disp["共著数"] = _disp["共著数"].fillna(0).astype(int)
                            _disp["相手著者数"] = _disp["相手著者数"].fillna(0).astype(int)
                            _disp = _disp.rename(columns={"cluster_img": "cluster"})

                            st.markdown(
                                "<div style='display:flex; align-items:center; gap:6px; margin:10px 0 4px 0;'>"
                                "<span style='font-weight:600; font-size:0.95rem; opacity:0.9;'>🧭 中心著者サマリー（cluster色連動）</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("<style>.stDataFrame [data-testid='stImage'] img { display:block; margin:auto; }</style>", unsafe_allow_html=True)
                            st.dataframe(
                                _disp,
                                column_config={
                                    "cluster": st.column_config.ImageColumn(
                                        "cluster",
                                        help="クラスタ色（ネットワークと同期）",
                                        width="small",
                                    ),
                                },
                                use_container_width=True,
                                hide_index=True,
                            )

                            _size_by_cluster = (
                                _merged.groupby("cluster_id").size().reset_index(name="size").sort_values("size", ascending=False)
                            )
                            _cid_to_rank = {int(cid): i + 1 for i, cid in enumerate(_size_by_cluster["cluster_id"].tolist())}

                            _legend_parts = [
                                "<div style='display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:6px 0 2px 0;'>",
                                "<span style='font-weight:700;'>クラスター凡例</span>",
                            ]
                            for _, rr in _size_by_cluster.merge(
                                _merged[["cluster_id", "cluster_color"]].drop_duplicates(), on="cluster_id", how="left"
                            ).iterrows():
                                rank_no = _cid_to_rank.get(int(rr["cluster_id"]), 0)
                                color = rr["cluster_color"]
                                count = int(rr["size"])
                                _legend_parts.append(
                                    f"<span style='display:inline-flex; align-items:center; gap:6px;'>"
                                    f"<span style='display:inline-block; width:12px; height:12px; background:{color}; border-radius:2px;'></span>"
                                    f"<span style='font-size:13px; opacity:0.9;'>C{rank_no}（{count}名）</span>"
                                    f"</span>"
                                )
                            _legend_parts.append("</div>")
                            st.markdown("".join(_legend_parts), unsafe_allow_html=True)

                            # 条件サマリー
                            st.caption(
                                "条件："
                                f"{_METRIC_JA.get(str(metric), str(metric))} ｜ "
                                f"ランキング件数：{int(top_n)} ｜ 最小共著回数：{int(min_w)} ｜ "
                                f"必須：{len(must_sel)}件／除外：{len(excl_sel)}件 ｜ "
                                + banners.summarize(GlobalFilters(y_from, y_to, tg_sel, tp_sel))
                            )
            except Exception as _e:
                st.caption(f"中心著者サマリーの生成に失敗しました: {_e!s}")
                st.caption(
                    "条件："
                    f"{_METRIC_JA.get(str(metric), str(metric))} ｜ "
                    f"ランキング件数：{int(top_n)} ｜ 最小共著回数：{int(min_w)} ｜ "
                    f"必須：{len(must_sel)}件／除外：{len(excl_sel)}件 ｜ "
                    + banners.summarize(GlobalFilters(y_from, y_to, tg_sel, tp_sel))
                )

            # ランキング表（上位）
            # st.dataframe(rank, use_container_width=True, hide_index=True)
            copyui.expander("📋 著者名をすぐコピー", rank["著者"].tolist())
            # ネットワーク描画（遅延）
            with st.expander("🕸️ ネットワークを可視化", expanded=False):
                vc1, vc2 = st.columns([1,1])
                with vc1:
                    top_only_cb = st.checkbox(
                        "上位ランキングの周辺だけ表示",
                        value=True,
                        key="res_net_toponly_cb",
                        help="上位に選ばれた著者本人と、その直接の共著者だけでサブグラフを作ります。"
                    )
                with vc2:
                    fixed_layout = st.checkbox(
                        "レイアウトを固定",
                        value=False,
                        key="res_net_fixed",
                        help="物理シミュレーションを止め、配置を固定します（位置がぶれません）。"
                    )
                if st.button("🌐 描画する", key="res_net_draw"):
                    top_nodes = rank["著者"].tolist() if top_only_cb else None
                    draw_network(
                        edges,
                        top_nodes=top_nodes,
                        min_weight=int(min_w),
                        height_px=700,
                        physics_enabled=(not fixed_layout),
                        node_color_map=node_color_map
                    )

    # ===== ③ トレンド分析 =====
    with tab_trend:
        yearly = yearly_author_counts(df_use)
        if yearly.empty:
            st.info("データがありません。")
            return
        tot = yearly.groupby("著者")["count"].sum().sort_values(ascending=False)
        options = tot.index.tolist()
        col_a, col_b, col_c = st.columns([1, 7, 1])
        with col_a:
            max_auth = st.number_input("初期表示数（上位）", min_value=3, max_value=30, value=10, step=1, key="res_trend_initn")
        default_sel = options[: int(max_auth)]
        with col_b:
            sel = st.multiselect("表示する著者（複数可）", options, default=default_sel, key="res_trend_authors")
        with col_c:
            ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="res_trend_ma")

        piv = yearly.pivot_table(index="発行年", columns="著者", values="count", aggfunc="sum").fillna(0).sort_index()
        if sel:
            piv = piv[[c for c in sel if c in piv.columns]]
        if piv.shape[1] == 0:
            st.info("表示対象がありません。")
            return
        if int(ma) > 1:
            piv = piv.rolling(window=int(ma), min_periods=1).mean()

        metric_mode = st.radio("表示指標", ["件数", "シェア(%)"], horizontal=True, key="res_trend_metric")
        if metric_mode == "シェア(%)":
            row_sums = piv.sum(axis=1)
            piv = piv.div(row_sums, axis=0).fillna(0) * 100

        if not piv.empty:
            try:
                last_row = piv.iloc[-1]
            except Exception:
                last_row = piv.mean(axis=0, numeric_only=True)
            order = list(last_row.sort_values(ascending=False).index)
            piv = piv.loc[:, [c for c in order if c in piv.columns]]

        try:
            import plotly.express as px
            fig = px.line(piv.reset_index().melt(id_vars="発行年", var_name="著者", value_name="値"),
                          x="発行年", y="値", color="著者", markers=True)
            fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), legend_title_text="著者",
                              yaxis_title=("件数" if metric_mode=="件数" else "シェア(%)"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.line_chart(piv)

        st.caption(f"条件：表示著者={len(sel)}名 ｜ 移動平均={int(ma)}年 ｜ 指標={metric_mode} ｜ " + _summarize(y_from, y_to, tg_sel, tp_sel))
        copyui.expander("📋 著者名をすぐコピー", list(piv.columns))

    # ⑥ 対象物別のTop5著者（改善版UI）: move inside tab_count, after caption and before copyui.expander
    with tab_count:
        # ... (existing code above remains unchanged)
        if s.empty:
            st.info("条件に合うデータがありません。")
        else:
            rank = s.reset_index()
            rank.columns = ["著者", "論文数"]
            rank = rank.sort_values(["論文数", "著者"], ascending=[False, True])
            rank_shown = rank.head(int(top_n))


            # ⑥ 対象物別のTop5著者（改善版UI）: only in 論文数サブタブ
            with st.expander("🏷️ 対象物別のTop5著者（現在のフィルタで集計）", expanded=False):
                # ▼ 見やすさ改善版：対象物ごとのTop5を「横棒グラフの小カード」で並べる（最大8グループ）
                view_mode = st.radio("表示形式", ["グラフ", "表"], horizontal=True, key="res_cnt_tg_view")
                try:
                    # 対象物ごとに著者カウント
                    rows = []
                    for _, r in df_rank.iterrows():
                        import re
                        split_multi = lambda s: [w.strip() for w in re.split(r"[;；,、，/／|｜\\s　]+", str(s or "")) if w.strip()]
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

            copyui.expander("📋 著者名をすぐコピー", rank_shown["著者"].tolist(), height=140)