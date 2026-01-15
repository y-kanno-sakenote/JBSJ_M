from __future__ import annotations
import pandas as pd
import streamlit as st
from .base import norm_key, short_preview, PALETTE, get_banner_filters
from .compute import build_keyword_cooccur_edges
from .network import compute_node_communities_from_edges, draw_pyvis_from_edges
from .copyui import expander as copy_expander

def render_cooccur_block(df_use: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns([1,1,1.6,1.6,0.9])
    with c2:
        min_edge = st.number_input("最低共起数（同時出現）", min_value=1, max_value=50, value=3, step=1, key="kw_co_minw")
    with c1:
        topN = st.number_input("表示キーワード数", min_value=30, max_value=300, value=120, step=10, key="kw_co_topn")
    with c3:
        include_raw = st.text_input("必須キーワード（部分一致・カンマ区切り）", value="", key="kw_co_include", placeholder="例: 酵母, 乳酸菌")
    with c4:
        exclude_raw = st.text_input("除外キーワード（部分一致・カンマ区切り）", value="", key="kw_co_exclude", placeholder="例: 試験, 実験")
    with c5:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        lcc_only = st.checkbox("主要ネットワークのみ", value=False, key="kw_co_lcc_only")

    include_list = [norm_key(x) for x in _split(include_raw)]
    exclude_list = [norm_key(x) for x in _split(exclude_raw)]

    edges = build_keyword_cooccur_edges(df_use, int(min_edge))
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
        deg = pd.concat([edges.groupby("src")["weight"].sum(), edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    if lcc_only:
        try:
            import networkx as nx
            if not edges.empty:
                G_tmp = nx.Graph()
                for _, r in edges.iterrows(): G_tmp.add_edge(str(r["src"]), str(r["dst"]))
                if G_tmp.number_of_nodes()>0:
                    comps = list(nx.connected_components(G_tmp))
                    if comps:
                        lcc_nodes = set(max(comps, key=len))
                        edges = edges[edges["src"].astype(str).isin(lcc_nodes) & edges["dst"].astype(str).isin(lcc_nodes)].reset_index(drop=True)
        except Exception as e:
            st.info(f"LCC 抽出で例外: {e}")

    st.caption(f"エッジ数: {len(edges)}")

    # Build and display co-occurrence edges table with cluster color and example titles
    comm_map = compute_node_communities_from_edges(edges)

    df_edges = edges.copy()
    def _edge_cluster_id(row):
        src_id = comm_map.get(str(row["src"]))
        dst_id = comm_map.get(str(row["dst"]))
        if src_id == dst_id and src_id is not None:
            return src_id
        if src_id is not None:
            return src_id
        if dst_id is not None:
            return dst_id
        return None
    df_edges["cluster_id"] = df_edges.apply(_edge_cluster_id, axis=1)

    cluster_img = df_edges["cluster_id"].map(lambda c: _color_square_data_uri(PALETTE[int(c) % len(PALETTE)]) if pd.notna(c) else "")
    df_edges["cluster_img"] = cluster_img

    df_edges = _attach_example_titles(df_use, df_edges, max_titles=3)

    show_df = df_edges[["cluster_img","src","dst","weight","example_titles"]].rename(columns={
        "cluster_img":"cluster",
        "src":"語A",
        "dst":"語B",
        "weight":"共起回数",
        "example_titles":"論文例"
    })

    st.dataframe(
        show_df.head(300),
        use_container_width=True,
        hide_index=True,
        column_config={
            "cluster": st.column_config.ImageColumn("cluster", help="自動クラスタ色（ネットワークと同期）。", width="small"),
            "語A": st.column_config.TextColumn("語A", width="small"),
            "語B": st.column_config.TextColumn("語B", width="small"),
            "共起回数": st.column_config.NumberColumn("共起回数", format="%d", width="small"),
            "論文例": st.column_config.TextColumn("論文例", help="そのペアが同時に登場した論文タイトルの例（最大3件）", width="large"),
        }
    )

    # クラスタ凡例（表でなく chips だけ）
    if comm_map:
        nodes_present = set(df_edges["src"].astype(str)).union(set(df_edges["dst"].astype(str)))
        counts = {}
        for n in nodes_present:
            cid = comm_map.get(str(n))
            if isinstance(cid, int): counts[cid] = counts.get(cid, 0) + 1
        chips = []
        for cid in sorted(counts.keys()):
            col = PALETTE[cid % len(PALETTE)]
            chips.append(f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{col};margin:0 6px 0 0;vertical-align:middle;'></span> C{cid+1}（{counts[cid]}語）")
        st.markdown("**クラスタ凡例**&nbsp;&nbsp;" + " ".join(chips), unsafe_allow_html=True)

    # 条件サマリー（期間の右に 対象物・研究タイプ を追加）
    _inc_pv = short_preview(include_list, 3)
    _exc_pv = short_preview(exclude_list, 3)
    y_from, y_to, tg_sel, tp_sel = get_banner_filters(prefix="kw")
    period = f"{int(y_from)}–{int(y_to)}" if (y_from is not None and y_to is not None) else "—"
    tg_preview = short_preview(tg_sel or [], 3)
    tp_preview = short_preview(tp_sel or [], 3)

    parts = [f"表示キーワード数{int(topN)}", f"最低共起数≧{int(min_edge)}"]
    if lcc_only:
        parts.append("主要ネットワークのみ")
    if _inc_pv:
        parts.append(f"必須：{_inc_pv}")
    if _exc_pv:
        parts.append(f"除外：{_exc_pv}")
    parts.append(f"期間：{period}")
    if tg_preview:
        parts.append(f"対象物：{tg_preview}")
    if tp_preview:
        parts.append(f"研究タイプ：{tp_preview}")
    st.caption(" ｜ ".join(parts))

    # コピーUI（ノード名）
    nodes = sorted(set(df_edges["src"].astype(str)).union(set(df_edges["dst"].astype(str)))) if not df_edges.empty else []
    copy_expander("📋 ノード名をすぐコピー", nodes)

    with st.expander("🕸️ ネットワークを可視化", expanded=False):
        freeze_layout = st.checkbox("レイアウトを固定", value=True, key="kw_co_freeze")
        if st.button("🌐 描画する", key="kw_co_draw"):
            draw_pyvis_from_edges(edges, height_px=680, freeze_layout=freeze_layout)
            # ネットワーク図の直下にも同じ条件サマリーを表示
            _inc_pv = short_preview(include_list, 3)
            _exc_pv = short_preview(exclude_list, 3)
            y_from, y_to, tg_sel, tp_sel = get_banner_filters(prefix="kw")
            period = f"{int(y_from)}–{int(y_to)}" if (y_from is not None and y_to is not None) else "—"
            tg_preview = short_preview(tg_sel or [], 3)
            tp_preview = short_preview(tp_sel or [], 3)
            _parts_draw = [f"表示キーワード数{int(topN)}", f"最低共起数≧{int(min_edge)}"]
            if lcc_only:
                _parts_draw.append("主要ネットワークのみ")
            if _inc_pv:
                _parts_draw.append(f"必須：{_inc_pv}")
            if _exc_pv:
                _parts_draw.append(f"除外：{_exc_pv}")
            _parts_draw.append(f"期間：{period}")
            if tg_preview:
                _parts_draw.append(f"対象物：{tg_preview}")
            if tp_preview:
                _parts_draw.append(f"研究タイプ：{tp_preview}")
            st.caption(" ｜ ".join(_parts_draw))

def _attach_example_titles(df_src: pd.DataFrame, edges: pd.DataFrame, max_titles: int = 3) -> pd.DataFrame:
    # Columns to try for titles
    title_columns = [
        "タイトル","論文タイトル","論文名","題名","title","Title","Japanese Title","English Title",
        "title_ja","title_en","タイトル（和）","タイトル（英）","和文タイトル","英文タイトル"
    ]
    available_cols = [col for col in title_columns if col in df_src.columns]

    # Columns to try for keywords
    keyword_columns = [
        "llm_keywords", "primary_keywords", "secondary_keywords", "featured_keywords"
    ] + [f"キーワード{i}" for i in range(1,11)]

    # Extract keywords sets from df_src rows
    # Use _split and norm_key for tokens
    def extract_keywords_from_row(row) -> set[str]:
        kw_set = set()
        for col in keyword_columns:
            if col in df_src.columns:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    kw_set.update(norm_key(k) for k in _split(val))
        return kw_set

    # Build list of (keyword_set, title)
    kw_title_list = []
    for _, row in df_src.iterrows():
        # Find first available title column with non-empty string
        title = None
        for col in available_cols:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                title = val.strip()
                break
        if not title:
            continue
        kw_set = extract_keywords_from_row(row)
        if not kw_set:
            continue
        kw_title_list.append((kw_set, title))

    # For each edge, find up to max_titles titles where both src and dst in kw_set
    example_titles = []
    for _, row in edges.iterrows():
        src = norm_key(str(row["src"]))
        dst = norm_key(str(row["dst"]))
        matched_titles = []
        for kw_set, title in kw_title_list:
            if src in kw_set and dst in kw_set:
                matched_titles.append(title)
                if len(matched_titles) >= max_titles:
                    break
        example_titles.append(" ／ ".join(matched_titles))
    edges = edges.copy()
    edges["example_titles"] = example_titles
    return edges

def _color_square_data_uri(hex_color: str, size: int = 12) -> str:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<rect width="{size}" height="{size}" fill="{hex_color}"/>
</svg>"""
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"

def _split(s: str) -> list[str]:
    import re
    return [w.strip() for w in re.split(r"[,;；、，/／|｜\s　]+", str(s or "")) if w.strip()]