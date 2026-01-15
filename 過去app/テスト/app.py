# -*- coding: utf-8 -*-
"""
論文検索UI（フォーム一括反映版）
- 左のフィルタで絞り込み
- 上：検索結果テーブル（★チェック可／フォーム内で一括反映）
- 下：お気に入り一覧（フィルタ無視で全体から表示／★チェック可／フォーム内で一括反映）
- HP/PDF はリンク化
- 「★」操作によるスクロールジャンプを防ぐため、st.form で一括反映
"""
import io, re, time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="論文検索（統一UI版）", layout="wide")

# ===== 列定義 =====
KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

# ===== ユーティリティ =====
def norm_space(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_key(s: str) -> str:
    return norm_space(s).lower()

AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
def split_authors(cell):
    if not cell: return []
    return [w.strip() for w in AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

def split_multi(s):
    if not s: return []
    return [w.strip() for w in re.split(r"[;；,、，/／|｜\s　]+", str(s)) if w.strip()]

def tokens_from_query(q):
    q = norm_key(q)
    return [t for t in re.split(r"[ ,，、；;　]+", q) if t]

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def consolidate_authors_column(df: pd.DataFrame) -> pd.DataFrame:
    """著者列：空白では分割しない。区切り記号のみで分割→セル内重複を代表表記に統合"""
    if "著者" not in df.columns:
        return df
    df = df.copy()
    def unify(cell: str) -> str:
        names = split_authors(cell)
        seen = set()
        result = []
        for n in names:
            k = norm_key(n)
            if not k or k in seen:
                continue
            seen.add(k)
            result.append(n)  # 先に出た表記を代表
        return ", ".join(result)
    df["著者"] = df["著者"].astype(str).apply(unify)
    return df

def build_author_candidates(df: pd.DataFrame):
    rep = {}
    for v in df.get("著者", pd.Series(dtype=str)).fillna(""):
        for name in split_authors(v):
            k = norm_key(name)
            if k and k not in rep:
                rep[k] = name
    return [rep[k] for k in sorted(rep.keys())]

def haystack(row, include_fulltext: bool):
    parts = [
        str(row.get("論文タイトル","")),
        str(row.get("著者","")),
        str(row.get("file_name","")),
        " ".join(str(row.get(c,"")) for c in KEY_COLS if c in row),
    ]
    if include_fulltext and "pdf_text" in row:
        parts.append(str(row.get("pdf_text","")))
    return norm_key(" \n ".join(parts))

def to_int_or_none(x):
    try:
        return int(str(x).strip())
    except Exception:
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None

def order_by_template(values, template):
    vs = list(dict.fromkeys(values))  # unique & keep order
    tmpl_set = set(template)
    head = [v for v in template if v in vs and "その他" not in v]
    mid  = sorted([v for v in vs if v not in tmpl_set and "その他" not in v])
    tail = [v for v in template if v in vs and "その他" in v] + \
           [v for v in vs if ("その他" in v and v not in template)]
    return head + mid + tail

def make_row_id(row):
    no = str(row.get("No.", "")).strip()
    if no and no.lower() not in {"none", "nan"}:
        return f"NO:{no}"
    ttl = str(row.get("論文タイトル", "")).strip()
    yr  = str(row.get("発行年", "")).strip()
    return f"T:{ttl}|Y:{yr}"

# ===== データ読み込み =====
st.title("論文検索（年・巻・号＋統一検索フィルタ）")

with st.sidebar:
    st.header("データ読み込み")
    url = st.text_input("公開CSVのURL（Googleスプレッドシート output=csv）", value="")
    up  = st.file_uploader("CSVをローカルから読み込み", type=["csv"])
    if st.button("読み込み", type="primary"):
        try:
            if up is not None:
                st.session_state.df = ensure_cols(pd.read_csv(up))
            elif url.strip():
                st.session_state.df = ensure_cols(fetch_csv(url.strip()))
            else:
                st.warning("URL または CSV を指定してください。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

df = st.session_state.get("df", pd.DataFrame())
if df.empty:
    st.info("左のサイドバーから CSV を指定して [読み込み] を押してください。")
    st.stop()

# No. が空の行は非表示
if "No." in df.columns:
    df = df[df["No."].apply(lambda v: str(v).strip() not in ("", "None", "nan"))]

# 著者表記の統合
df = consolidate_authors_column(df)

# ===== 年・巻・号（1行） =====
st.subheader("年・巻・号フィルタ")

year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
if year_vals.notna().any():
    ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
else:
    ymin_all, ymax_all = 1980, 2025

c_y, c_v, c_i = st.columns([1, 1, 1])
with c_y:
    y_from, y_to = st.slider(
        "発行年（範囲）", min_value=ymin_all, max_value=ymax_all,
        value=(ymin_all, ymax_all)  # 全範囲を初期値に
    )
with c_v:
    vol_candidates = sorted({v for v in (df.get("巻数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    vols_sel = st.multiselect("巻（整数・複数選択）", vol_candidates, default=[])
with c_i:
    iss_candidates = sorted({v for v in (df.get("号数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    issues_sel = st.multiselect("号（整数・複数選択）", iss_candidates, default=[])

# ===== 著者・対象物・研究タイプ・キーワード =====
st.subheader("統一検索フィルタ")

c_a, c_tg, c_tp = st.columns([1.2, 1.2, 1.2])
with c_a:
    authors_all = build_author_candidates(df)
    authors_sel = st.multiselect("著者（正規化＋個別）", authors_all, default=[])

with c_tg:
    raw_targets = {t for v in df.get("対象物", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
    targets_sel = st.multiselect("対象物（複数選択／部分一致）", targets_all, default=[])

with c_tp:
    raw_types = {t for v in df.get("研究タイプ", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    types_all = order_by_template(list(raw_types), TYPE_ORDER)
    types_sel = st.multiselect("研究タイプ（複数選択／部分一致）", types_all, default=[])

c_kw1, c_kw2, c_kw3 = st.columns([3, 1, 1])
with c_kw1:
    kw_query = st.text_input("キーワード（空白/カンマで複数可）", value="")
with c_kw2:
    kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True)
with c_kw3:
    include_fulltext = st.checkbox("本文も検索（pdf_text）", value=True)

# ===== フィルタ適用 =====
def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()

    # 年
    if "発行年" in df2.columns:
        y = pd.to_numeric(df2["発行年"], errors="coerce")
        df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]

    # 巻・号
    if vols_sel and "巻数" in df2.columns:
        df2 = df2[df2["巻数"].map(to_int_or_none).isin(set(vols_sel))]
    if issues_sel and "号数" in df2.columns:
        df2 = df2[df2["号数"].map(to_int_or_none).isin(set(issues_sel))]

    # 著者（空白で分割しない）
    if authors_sel and "著者" in df2.columns:
        sel = {norm_key(a) for a in authors_sel}
        def hit_author(v):
            return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]

    # 対象物 / 研究タイプ（部分一致：OR）
    if targets_sel and "対象物" in df2.columns:
        t_norm = [norm_key(t) for t in targets_sel]
        df2 = df2[df2["対象物"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    if types_sel and "研究タイプ" in df2.columns:
        t_norm = [norm_key(t) for t in types_sel]
        df2 = df2[df2["研究タイプ"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]

    # キーワード
    toks = tokens_from_query(kw_query)
    if toks:
        def hit_kw(row):
            hs = haystack(row, include_fulltext=include_fulltext)
            return all(t in hs for t in toks) if kw_mode == "AND" else any(t in hs for t in toks)
        df2 = df2[df2.apply(hit_kw, axis=1)]
    return df2

filtered = apply_filters(df)

# ===== 表の見た目制御（隠し列・リンク化など） =====
st.markdown("### 検索結果")
st.caption(f"{len(filtered)} / {len(df)} 件")

# 非表示列：相対PASS、終了ページ、file_path、num_pages、file_name、llm_keywords以降すべて
all_cols = list(filtered.columns)
hide_cols = {"相対PASS", "終了ページ", "file_path", "num_pages", "file_name"}
if "llm_keywords" in all_cols:
    start = all_cols.index("llm_keywords")
    hide_cols.update(all_cols[start:])  # llm_keywords 以降を非表示
visible_cols = [c for c in all_cols if c not in hide_cols]

# 右（お気に入り）側はフィルタ無視の全体可視列
all_cols_full = list(df.columns)
hide_cols_full = {"相対PASS", "終了ページ", "file_path", "num_pages", "file_name"}
if "llm_keywords" in all_cols_full:
    start_full = all_cols_full.index("llm_keywords")
    hide_cols_full.update(all_cols_full[start_full:])
visible_cols_full = [c for c in all_cols_full if c not in hide_cols_full]

# 可視データ
disp = filtered[visible_cols].copy()

# 一意ID（お気に入り管理用）
disp["_row_id"] = disp.apply(make_row_id, axis=1)

# セッションにお気に入り集合
if "favs" not in st.session_state:
    st.session_state.favs = set()

# 現在のお気に入り反映（★初期値）
disp["★"] = disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)

# LinkColumn 設定
column_config_main = {
    "★": st.column_config.CheckboxColumn("★", help="気になる論文にチェック/解除", default=False, width="small"),
}
if "HPリンク先" in disp.columns:
    column_config_main["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", help="外部サイトへ移動", display_text="HP")
if "PDFリンク先" in disp.columns:
    column_config_main["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", help="PDFを開く", display_text="PDF")

display_order = ["★"] + [c for c in disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

# ===== 上：メイン表（フォーム一括反映） =====
st.subheader("全件（編集可）")
with st.form("main_form", clear_on_submit=False):
    edited_main = st.data_editor(
        disp[display_order],
        key="main_editor",
        use_container_width=True,
        hide_index=True,
        column_config=column_config_main,
        disabled=[c for c in display_order if c != "★"],  # ★のみ編集可
        height=520,
        num_rows="fixed",
    )
    submitted_main = st.form_submit_button("✅ 変更を反映（上の表）", use_container_width=True)

# 上フォームの反映ロジック
if submitted_main:
    subset_ids_main = set(disp["_row_id"].tolist())
    checked_subset_main = set(edited_main.loc[edited_main["★"] == True, "_row_id"].tolist())
    st.session_state.favs = (st.session_state.favs - subset_ids_main) | checked_subset_main
    st.rerun()

st.divider()

# ===== 下：お気に入り一覧（フォーム一括反映／フィルタ無視） =====
st.subheader(f"⭐ お気に入り一覧（常設） — 現在 {len(st.session_state.favs)} 件")

fav_disp_full = df[visible_cols_full].copy()
fav_disp_full["_row_id"] = fav_disp_full.apply(make_row_id, axis=1)
fav_disp = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

# サブ表：データが無い場合の案内
if fav_disp.empty:
    st.info("お気に入りは未選択です。上の表の『★』にチェックしてください。")
else:
    fav_disp["★"] = fav_disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)
    fav_display_order = ["★"] + [c for c in fav_disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

    column_config_fav = {
        "★": st.column_config.CheckboxColumn("★", help="チェックで解除/追加", default=True, width="small"),
    }
    if "HPリンク先" in fav_disp.columns:
        column_config_fav["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", display_text="HP")
    if "PDFリンク先" in fav_disp.columns:
        column_config_fav["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", display_text="PDF")

    # 右端に「全て外す」ボタン
    c1, c2 = st.columns([1, 5])
    with c2:
        st.write("")  # 位置調整
    with c1:
        if st.button("❌ 全て外す", use_container_width=True):
            st.session_state.favs = set()
            st.rerun()

    with st.form("fav_form", clear_on_submit=False):
        fav_edited = st.data_editor(
            fav_disp[fav_display_order],
            key="fav_editor",
            use_container_width=True,
            hide_index=True,
            column_config=column_config_fav,
            disabled=[c for c in fav_display_order if c != "★"],
            height=420,
            num_rows="fixed",
        )
        submitted_fav = st.form_submit_button("✅ 変更を反映（下の表）", use_container_width=True)

    if submitted_fav:
        subset_ids_fav = set(fav_disp["_row_id"].tolist())
        fav_checked_subset = set(fav_edited.loc[fav_edited["★"] == True, "_row_id"].tolist())
        st.session_state.favs = (st.session_state.favs - subset_ids_fav) | fav_checked_subset
        st.rerun()

# ===== エクスポート（表示列のみ） =====
export_df = edited_main.drop(columns=["★", "_row_id"])
st.download_button(
    "📥 絞り込み結果をCSV出力（表示列のみ）",
    data=export_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"filtered_{time.strftime('%Y%m%d')}.csv",
    mime="text/csv",
    use_container_width=True
)