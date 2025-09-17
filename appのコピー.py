# -*- coding: utf-8 -*-
import io, re, time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="論文検索（統一UI版）", layout="wide")

KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]
BASE_COLS = [
    "No.","相対PASS","発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "対象物","研究タイプ",
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
]

# ---- 並び順（存在するものだけこの順で表示） ----
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

# ---------- utils ----------
def norm_space(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00A0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_key(s: str) -> str:
    return norm_space(s).lower()

# 著者の分割：※空白では分割しない（姓と名を保持）
AUTHOR_SPLIT_RE = re.compile(r"[;；,、，/／|｜]+")
def split_authors(cell):
    if not cell: return []
    return [w.strip() for w in AUTHOR_SPLIT_RE.split(str(cell)) if w.strip()]

# 一般キーワードの分割（空白もOK）
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
    """著者列：空白で分割しない。区切り記号のみで分割→セル内重複を代表表記に統合"""
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
    # 1) テンプレにあるものを順に 2) 残りをアルファ順 3) テンプレにある「その他系」は最後
    tmpl_set = set(template)
    head = [v for v in template if v in vs and "その他" not in v]
    mid  = sorted([v for v in vs if v not in tmpl_set and "その他" not in v])
    tail = [v for v in template if v in vs and "その他" in v] + \
           [v for v in vs if ("その他" in v and v not in template)]
    return head + mid + tail

# ---------- load ----------
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

# No. が None/空の行は非表示
if "No." in df.columns:
    df = df[df["No."].apply(lambda v: str(v).strip() not in ("", "None", "nan"))]

# 著者セルの異表記統合（※空白で分割しない）
df = consolidate_authors_column(df)

# ---------- 年・巻・号（1行） ----------
st.subheader("年・巻・号フィルタ")

year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
if year_vals.notna().any():
    ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
    default_from = max(ymin_all, ymax_all - 19)  # 直近20年
    default_to   = ymax_all
else:
    ymin_all, ymax_all = 1980, 2025
    default_from, default_to = 2006, 2025

c_y, c_v, c_i = st.columns([1, 1, 1])
with c_y:
    y_from, y_to = st.slider(
        "発行年（範囲）",
        min_value=ymin_all, max_value=ymax_all,
        value=(ymin_all, ymax_all)  # ★ 全範囲を初期値に
    )
with c_v:
    vol_candidates = sorted({v for v in (df.get("巻数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    vols_sel = st.multiselect("巻（整数・複数選択）", vol_candidates, default=[])
with c_i:
    iss_candidates = sorted({v for v in (df.get("号数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    issues_sel = st.multiselect("号（整数・複数選択）", iss_candidates, default=[])

# ---------- 著者・対象物・研究タイプ（1行） ----------
st.subheader("統一検索フィルタ")

c_a, c_tg, c_tp = st.columns([1.2, 1.2, 1.2])
with c_a:
    authors_all = build_author_candidates(df)
    authors_sel = st.multiselect("著者（正規化＋個別）", authors_all, default=[])

with c_tg:
    # セル内複数値を集約 → 指定順に並べ替え
    raw_targets = {t for v in df.get("対象物", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
    targets_sel = st.multiselect("対象物（複数選択／部分一致）", targets_all, default=[])

with c_tp:
    raw_types = {t for v in df.get("研究タイプ", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    types_all = order_by_template(list(raw_types), TYPE_ORDER)
    types_sel = st.multiselect("研究タイプ（複数選択／部分一致）", types_all, default=[])

# キーワード
c_kw1, c_kw2, c_kw3 = st.columns([3, 1, 1])
with c_kw1:
    kw_query = st.text_input("キーワード（空白/カンマで複数可）", value="")
with c_kw2:
    kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True)
with c_kw3:
    include_fulltext = st.checkbox("本文も検索（pdf_text）", value=True)

# ---------- フィルタ適用 ----------
def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()

    # 年
    if "発行年" in df2.columns:
        y = pd.to_numeric(df2["発行年"], errors="coerce")
        df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]

    # 巻・号（整数）
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

# -------------------------------------------------
# 結果表示・DL（表示用カラム制御 & リンク化）
# -------------------------------------------------
st.markdown("### 検索結果")
st.caption(f"{len(filtered)} / {len(df)} 件")

all_cols = list(filtered.columns)

# 非表示ターゲット
hide_cols = {"相対PASS", "終了ページ", "file_path", "num_pages", "file_name"}
if "llm_keywords" in all_cols:
    start = all_cols.index("llm_keywords")
    hide_cols.update(all_cols[start:])  # llm_keywords 以降を全て非表示

visible_cols = [c for c in all_cols if c not in hide_cols]

# DataFrame（リンク化のカラム設定）
column_config = {}
if "HPリンク先" in visible_cols:
    column_config["HPリンク先"] = st.column_config.LinkColumn(
        "HPリンク先", help="外部サイトへ移動", display_text="HP"
    )
if "PDFリンク先" in visible_cols:
    column_config["PDFリンク先"] = st.column_config.LinkColumn(
        "PDFリンク先", help="PDFを開く", display_text="PDF"
    )

st.dataframe(
    filtered[visible_cols],
    use_container_width=True,
    hide_index=True,
    column_config=column_config
)

# 画面と同じ列だけエクスポート
csv_bytes = filtered[visible_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "絞り込み結果をCSV出力（表示列のみ）",
    data=csv_bytes,
    file_name=f"filtered_{time.strftime('%Y%m%d')}.csv",
    mime="text/csv"
)