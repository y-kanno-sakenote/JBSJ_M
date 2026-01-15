# -*- coding: utf-8 -*-
"""
論文検索（統一UI版：お気に入りにタグを“表で直接入力”）

機能:
- 発行年レンジ、巻・号（複数選択）、著者（正規化・複数選択）、対象物/研究タイプ（部分一致・複数選択）
- キーワード AND/OR 検索（空白/カンマ区切り、pdf_text を含めるか選択可能）
- 検索結果テーブル（不要列の非表示、HP/PDF のリンク化、★でお気に入り）
- お気に入り一覧（常設・★で解除/追加）
- お気に入りタグ付け：お気に入り表の「tags」列を直接編集（カンマ/空白区切り）
- 「❌ 全て外す」ボタンでお気に入り一括解除
"""

import io, re, time
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

# -------------------- ページ設定 --------------------
st.set_page_config(page_title="論文検索（統一UI版）", layout="wide")


# -------------------- コントラスト（著者ドロップダウン強化版） --------------------
st.markdown(
    """
    <style>
    /* ========= テキスト入力欄（枠線あり） ========= */
    .stTextInput input, .stNumberInput input, textarea {
      background-color: #e0e0e0 !important;
      color: #000 !important;
      border: 1px solid #666 !important;
      border-radius: 6px !important;
      padding: 4px 8px !important;
    }

    /* ========= Select / MultiSelect（外枠デザイン） ========= */
    .stMultiSelect div[data-baseweb="select"],
    .stSelectbox  div[data-baseweb="select"] {
      background-color: #e0e0e0 !important;
      border: 1px solid #666 !important;
      border-radius: 6px !important;
    }
    div[data-baseweb="select"] > div { background: transparent !important; }
    div[data-baseweb="select"] span { color: #000 !important; }
    div[data-baseweb="select"] svg  { color: #000 !important; fill: #000 !important; }

    /* MultiSelect のタグ */
    div[data-baseweb="tag"] {
      background: #d5d5d5 !important;
      color: #000 !important;
      border-radius: 12px !important;
    }

    /* --- フォーカス時のスタイル --- */
    input:focus, textarea:focus,
    .stTextInput input:focus, .stNumberInput input:focus {
      border: 2px solid #1a73e8 !important;
      box-shadow: 0 0 4px #1a73e8 !important;
      outline: none !important;
    }
    .stMultiSelect div[data-baseweb="select"]:focus-within,
    .stSelectbox  div[data-baseweb="select"]:focus-within {
      border: 2px solid #1a73e8 !important;
      box-shadow: 0 0 4px #1a73e8 !important;
    }
    .stMultiSelect input:focus,
    .stSelectbox  input:focus {
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
    }

    /* ======== ドロップダウン（候補リスト）を“高く & 太いスクロール”に ======== */
    ul[role="listbox"] {
      background: #f5f5f5 !important;
      border: 1px solid #666 !important;
      max-height: 70vh !important;     /* 画面の7割まで高く */
      min-height: 360px !important;     /* 低解像度でも十分な高さを確保 */
      overflow-y: auto !important;
      padding-right: 6px !important;    /* スクロールバーぶんの余白 */
      scrollbar-width: auto;            /* Firefox: 太め */
      scrollbar-color: #555 #e9e9e9;    /* Firefox: コントラスト強め */
    }
    /* 候補1行を少し高く＝当たり判定を大きくして選びやすく */
    li[role="option"] {
      padding: 8px 12px !important;
      line-height: 1.4 !important;
      font-size: 0.95rem !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
      background: #e0e0e0 !important;
      color: #000 !important;
    }

    /* WebKit 系（Chrome/Edge/Safari）のスクロールバーを太く＆高コントラストに */
    ul[role="listbox"]::-webkit-scrollbar {
      width: 16px;                      /* ← 太くする */
    }
    ul[role="listbox"]::-webkit-scrollbar-track {
      background: #e9e9e9;
      border-radius: 8px;
    }
    ul[role="listbox"]::-webkit-scrollbar-thumb {
      background: #555;
      border-radius: 8px;
      border: 3px solid #e9e9e9;        /* 外側に余白をとって“太く見せる” */
    }
    ul[role="listbox"]::-webkit-scrollbar-thumb:hover {
      background: #333;
    }
    </style>
    """,
    unsafe_allow_html=True

)

# -------------------- 定数 --------------------
KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]
BASE_COLS = [
    "No.","相対PASS","発行年","巻数","号数","開始ページ","終了ページ",
    "論文タイトル","著者","file_name","HPリンク先","PDFリンク先",
    "対象物_top3","研究タイプ_top3",
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
]
TARGET_ORDER = [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = [
    "微生物・遺伝子関連","醸造工程・製造技術","応用利用・食品開発","成分分析・物性評価",
    "品質評価・官能評価","歴史・文化・経済","健康機能・栄養効果","統計解析・モデル化",
    "環境・サステナビリティ","保存・安定性","その他（研究タイプ）"
]

# -------------------- ユーティリティ --------------------
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
    """著者列：空白では分割せず、区切り記号のみで分割→同名異表記を代表表記に統合"""
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
            result.append(n)
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

def haystack(row):
    parts = [
        str(row.get("論文タイトル","")),
        str(row.get("著者","")),
        str(row.get("file_name","")),
        " ".join(str(row.get(c,"")) for c in KEY_COLS if c in row),
    ]
    return norm_key(" \n ".join(parts))

def to_int_or_none(x):
    try: return int(str(x).strip())
    except Exception:
        m = re.search(r"\d+", str(x))
        return int(m.group()) if m else None

def order_by_template(values, template):
    """1) テンプレの順 2) 未収載はアルファ順 3) その他は最後"""
    vs = list(dict.fromkeys(values))
    tmpl_set = set(template)
    head = [v for v in template if v in vs and "その他" not in v]
    mid  = sorted([v for v in vs if v not in tmpl_set and "その他" not in v])
    tail = [v for v in template if v in vs and "その他" in v] + \
           [v for v in vs if ("その他" in v and v not in template)]
    return head + mid + tail

def make_visible_cols(df: pd.DataFrame) -> list[str]:
    """df の列から『相対PASS/終了ページ/file_path/num_pages/file_name』と
       『llm_keywords 以降の全列』を非表示対象にして、表示列リストを返す。
    """
    base_hide = {"相対PASS", "終了ページ", "file_path", "num_pages", "file_name"}
    cols = [str(c) for c in df.columns]
    hide = set(c for c in cols if c in base_hide)
    if "llm_keywords" in cols:
        idx = cols.index("llm_keywords")
        hide.update(cols[idx:])
    return [c for c in cols if c not in hide]

def make_row_id(row):
    no = str(row.get("No.", "")).strip()
    if no and no.lower() not in {"none", "nan"}:
        return f"NO:{no}"
    ttl = str(row.get("論文タイトル", "")).strip()
    yr  = str(row.get("発行年", "")).strip()
    return f"T:{ttl}|Y:{yr}"

# -------------------- データ読み込み --------------------
st.title("醸造協会誌　論文検索")

DEMO_CSV_PATH = Path("data/keywords_summary5.csv")   # メインCSV
SUMMARY_CSV_PATH = Path("data/summaries.csv")         # ← 追加: summary
AUTHORS_CSV_PATH = Path("data/authors_readings.csv")  # ← 追加: 著者読み

SECRET_URL = st.secrets.get("GSHEET_CSV_URL", "")  # （任意）Secretsに入れておけば自動使用

@st.cache_data(ttl=600, show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(path, encoding="utf-8"))

@st.cache_data(ttl=600, show_spinner=False)
def load_url_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")

# --- 追加：summaries.csv ローダ ---
@st.cache_data(ttl=600, show_spinner=False)
def load_summaries(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df_s = pd.read_csv(path, encoding="utf-8")
        df_s.columns = [str(c).strip() for c in df_s.columns]
        if not {"file_name", "summary"}.issubset(df_s.columns):
            return None
        return df_s[["file_name", "summary"]]
    except Exception:
        return None

# --- 追加：authors_readings.csv ローダ ---
@st.cache_data(ttl=600, show_spinner=False)
def load_authors_readings(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        adf = pd.read_csv(path, encoding="utf-8")
        adf.columns = [str(c).strip() for c in adf.columns]
        if not {"author", "reading"}.issubset(adf.columns):
            return None
        adf["author"]  = adf["author"].astype(str).str.strip()
        adf["reading"] = adf["reading"].astype(str).str.strip()
        adf = adf[(adf["author"]!="") & (adf["reading"]!="")]
        # 同じ reading が複数ある場合は先勝ち
        adf = adf.drop_duplicates(subset=["reading"], keep="first")
        return adf
    except Exception:
        return None

with st.sidebar:
    st.header("データ読み込み")
    st.caption("※ まずはデモ用CSVを自動ロード。URL/ファイル指定で上書きできます。")

    # デモ自動ロードのON/OFF（デフォルトON）
    use_demo = st.toggle("デモCSVを自動ロードする", value=True, help="data/demo.csv を読み込みます。")

    # 上書き手段：URL or ファイル
    url = st.text_input("公開CSVのURL（Googleスプレッドシート output=csv）", value=SECRET_URL)
    up  = st.file_uploader("CSVをローカルから読み込み", type=["csv"])

    # 明示ボタン：読み込み（URL/ファイルの優先度は「アップロード > URL」）
    load_clicked = st.button("読み込み（URL/ファイルを優先）", type="primary", key="load_btn")

# 優先順位: 1) クリックでURL/ファイル 2) デモ自動 3) 最後の手段：待機
df = None
err = None

try:
    if load_clicked:
        if up is not None:
            df = ensure_cols(pd.read_csv(up, encoding="utf-8"))
            st.toast("ローカルCSVを読み込みました")
        elif url.strip():
            df = load_url_csv(url.strip())
            st.toast("URLのCSVを読み込みました")
        else:
            st.warning("URL または CSV を指定してください。")
    elif use_demo and DEMO_CSV_PATH.exists():
        df = load_local_csv(DEMO_CSV_PATH)
        st.caption(f"✅ デモCSVを自動ロード中: {DEMO_CSV_PATH}")
    elif SECRET_URL:
        df = load_url_csv(SECRET_URL)
        st.caption("✅ SecretsのURLから自動ロード中")
except Exception as e:
    err = e

if df is None:
    if err:
        st.error(f"読み込みエラー: {err}")
    st.info("左のサイドバーで CSV を指定するか、デモCSVを有効にしてください。")
    st.stop()

# --- 追加：summary をマージ ---
sum_df = load_summaries(SUMMARY_CSV_PATH)
if sum_df is not None:
    df = df.merge(sum_df, on="file_name", how="left")

# -------------------- 年・巻・号フィルタ --------------------
st.subheader("検索フィルタ")
year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
if year_vals.notna().any():
    ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
else:
    ymin_all, ymax_all = 1980, 2025

c_y, c_v, c_i = st.columns([1, 1, 1])
with c_y:
    y_from, y_to = st.slider(
        "発行年（範囲）", min_value=ymin_all, max_value=ymax_all,
        value=(ymin_all, ymax_all)
    )
with c_v:
    vol_candidates = sorted({v for v in (df.get("巻数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    vols_sel = st.multiselect("巻（複数選択）", vol_candidates, default=[])
with c_i:
    iss_candidates = sorted({v for v in (df.get("号数", pd.Series(dtype=str)).map(to_int_or_none)).dropna().unique()})
    issues_sel = st.multiselect("号（複数選択）", iss_candidates, default=[])

# -------------------- 検索フィルタ（1段目：対象物 / 研究タイプ） --------------------
#st.subheader("検索フィルタ")

row1_tg, row1_tp = st.columns([1.2, 1.2])

with row1_tg:
    raw_targets = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
    targets_sel = st.multiselect("対象物（複数選択／部分一致）", targets_all, default=[])

with row1_tp:
    raw_types = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
    types_all = order_by_template(list(raw_types), TYPE_ORDER)
    types_sel = st.multiselect("研究タイプ（複数選択／部分一致）", types_all, default=[])

# -------------------- 検索フィルタ（2段目：著者 + イニシャルラジオ横並び） --------------------
if "authors_sel" not in st.session_state:
    st.session_state.authors_sel = []

def filter_by_author(author_name):
    st.session_state.authors_sel = [author_name]
    st.rerun()

def handle_author_multiselect_change():
    # on_changeコールバックでは、`st.session_state`のキーを直接参照
    selected_readings = st.session_state.authors_multiselect_key
    
    reading2author = dict(zip(st.session_state.author_candidates["reading"], st.session_state.author_candidates["author"]))
    st.session_state.authors_sel = sorted({reading2author[r] for r in selected_readings}) if selected_readings else []
    
    # 著者フィルターの変更は rerunn が必要
    st.rerun()

row2_author, row2_radio = st.columns([1.0, 2.0])   # ← 著者欄を短めにしてラジオに幅を多めに

with row2_radio:
    initials = ["すべて","あ","か","さ","た","な","は","ま","や","ら","わ","英字"]
    if "author_initial" not in st.session_state:
        st.session_state.author_initial = "すべて"
    st.radio(
    "著者イニシャル選択",
    options=initials,
    horizontal=True,
    key="author_initial",   # ← これが唯一のソースオブトゥルース
)

# 以降は session_state から読むだけ（代入しない）
ini = st.session_state["author_initial"]
# authors_readings.csv を読み込み
if "author_candidates" not in st.session_state:
    st.session_state.author_candidates = load_authors_readings(AUTHORS_CSV_PATH)
with row2_author:
    adf = st.session_state.author_candidates
    if adf is not None and not adf.empty:
        cand = adf.copy()

        # --- （以下は従来と同じフィルタ＆並び替え処理）---
        GOJUON = {
            "あ": "あいうえお",
            "か": "かきくけこがぎぐげご",
            "さ": "さしすせそざじずぜぞ",
            "た": "たちつてとだぢづでど",
            "な": "なにぬねの",
            "は": "はひふへほばびぶべぼぱぴぷぺぽ",
            "ま": "まみむめも",
            "や": "やゆよ",
            "ら": "らりるれろ",
            "わ": "わをん",
        }

        def kata_to_hira(s: str) -> str:
            out = []
            for ch in str(s or ""):
                o = ord(ch)
                if 0x30A1 <= o <= 0x30F6:
                    out.append(chr(o - 0x60))
                else:
                    out.append(ch)
            return "".join(out)

        def hira_head(s: str) -> str | None:
            s = str(s or "")
            return kata_to_hira(s)[0] if s else None

        def is_roman_head(s: str) -> bool:
            return bool(re.match(r"[A-Za-z]", str(s or "")))

        ini = st.session_state.author_initial
        if ini == "英字":
            cand = cand[cand["reading"].astype(str).str.match(r"[A-Za-z]")]
        elif ini != "すべて":
            allowed = set(GOJUON.get(ini, ""))
            cand = cand[cand["reading"].apply(
                lambda s: (not is_roman_head(s)) and (hira_head(s) in allowed if hira_head(s) else False)
            )]

        # 並び順
        AIUEO_ORDER = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
        def sort_tuple(reading: str):
            if not reading: return (3, 999, "")
            ch = reading[0]
            if re.match(r"[A-Za-z]", ch): return (2, 999, ch.lower())
            if re.match(r"[\u30A0-\u30FF]", ch): return (1, 999, reading)
            idx = AIUEO_ORDER.find(ch)
            return (0, idx if idx != -1 else 998, reading)

        cand = cand.assign(
            _grp=[sort_tuple(r)[0] for r in cand["reading"]],
            _key=[sort_tuple(r)[1] for r in cand["reading"]],
            _sub=[sort_tuple(r)[2] for r in cand["reading"]],
        ).sort_values(by=["_grp","_key","_sub"], kind="mergesort").drop(columns=["_grp","_key","_sub"])

        reading2author = dict(zip(cand["reading"], cand["author"]))
        options_readings = list(reading2author.keys())
        
        selected_readings = []
        if 'authors_sel' in st.session_state:
            selected_author_names = set(st.session_state.authors_sel)
            for r, a in reading2author.items():
                if a in selected_author_names:
                    selected_readings.append(r)

        authors_sel_readings = st.multiselect(
            "著者（読みで検索可 / 表示は漢字＋読み）",
            options=options_readings,
            default=selected_readings,
            format_func=lambda r: f"{reading2author.get(r, r)}｜{r}",
            placeholder="例：やまだ / さとう / たかはし ...",
            on_change=handle_author_multiselect_change,
            key="authors_multiselect_key"
        )
    else:
        authors_all = build_author_candidates(df)
        st.session_state.authors_sel = st.multiselect("著者", authors_all, default=st.session_state.authors_sel)


# -------------------- 検索フィルタ（3段目：キーワード） --------------------
kw_row1, kw_row2 = st.columns([3, 1])
with kw_row1:
    kw_query = st.text_input("キーワード（空白/カンマで複数可）", value="")
with kw_row2:
    kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="kw_mode")


# -------------------- フィルタ適用 --------------------
def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
    df2 = _df.copy()
    if "発行年" in df2.columns:
        y = pd.to_numeric(df2["発行年"], errors="coerce")
        df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]
    if vols_sel and "巻数" in df2.columns:
        df2 = df2[df2["巻数"].map(to_int_or_none).isin(set(vols_sel))]
    if issues_sel and "号数" in df2.columns:
        df2 = df2[df2["号数"].map(to_int_or_none).isin(set(issues_sel))]
    if st.session_state.authors_sel and "著者" in df2.columns:
        sel = {norm_key(a) for a in st.session_state.authors_sel}
        def hit_author(v): return any(norm_key(x) in sel for x in split_authors(v))
        df2 = df2[df2["著者"].apply(hit_author)]
    if targets_sel and "対象物_top3" in df2.columns:
        t_norm = [norm_key(t) for t in targets_sel]
        df2 = df2[df2["対象物_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    if types_sel and "研究タイプ_top3" in df2.columns:
        t_norm = [norm_key(t) for t in types_sel]
        df2 = df2[df2["研究タイプ_top3"].apply(lambda v: any(t in norm_key(v) for t in t_norm))]
    toks = tokens_from_query(kw_query)
    if toks:
        def hit_kw(row):
            hs = haystack(row)
            return all(t in hs for t in toks) if kw_mode == "AND" else any(t in hs for t in toks)
        df2 = df2[df2.apply(hit_kw, axis=1)]
    return df2

filtered = apply_filters(df)

# -------------------- 検索結果テーブル --------------------
st.markdown("### 検索結果")
st.caption(f"{len(filtered)} / {len(df)} 件")

visible_cols = make_visible_cols(filtered)

# ★ ここで summary の位置を調整（著者の右に挿入）
if "著者" in visible_cols and "summary" in filtered.columns:
    idx = visible_cols.index("著者")
    if "summary" not in visible_cols:
        visible_cols.insert(idx + 1, "summary")

disp = filtered.loc[:, visible_cols].copy()
disp["_row_id"] = disp.apply(make_row_id, axis=1)

# セッション初期化：お気に入り集合／タグ辞書
if "favs" not in st.session_state:
    st.session_state.favs = set()
if "fav_tags" not in st.session_state:
    st.session_state.fav_tags = {}   # row_id -> set(tags)

# メイン表：お気に入りチェック列
disp["★"] = disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)


def handle_main_editor_change():
    edited_rows_dict = st.session_state.main_editor['edited_rows']
    for row_index_str, changes in edited_rows_dict.items():
        row_index = int(row_index_str)
        row_id = disp.iloc[row_index]['_row_id']
        if '★' in changes:
            if changes['★']:
                st.session_state.favs.add(row_id)
            else:
                st.session_state.favs.discard(row_id)
    # on_changeで直接stateを変更すればrerunは不要
    # ただし、他のウィジェットの変更を即座に反映させる場合はrerunが必要

def update_fav_and_tags_from_favs():
    edited_favs = st.session_state.fav_editor['edited_rows']
    fav_ids_in_view = fav_disp['_row_id'].tolist()
    
    for row_index_str, changes in edited_favs.items():
        row_index = int(row_index_str)
        row_id = fav_ids_in_view[row_index]

        # ★の変更を処理
        if '★' in changes:
            if changes['★']:
                st.session_state.favs.add(row_id)
            else:
                st.session_state.favs.discard(row_id)

        # tagsの変更を処理
        if 'tags' in changes:
            tag_set = {t.strip() for t in re.split(r"[ ,，、；;　]+", str(changes['tags'])) if t.strip()}
            if tag_set:
                st.session_state.fav_tags[row_id] = tag_set
            elif row_id in st.session_state.fav_tags:
                del st.session_state.fav_tags[row_id]
                
    st.rerun()


# --- メイン表 ---
st.subheader("論文リスト")

column_config = {
    "★": st.column_config.CheckboxColumn("★", help="気になる論文にチェック/解除", default=False, width="small"),
}
if "HPリンク先" in disp.columns:
    column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", help="外部サイトへ移動", display_text="HP")
if "PDFリンク先" in disp.columns:
    column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", help="PDFを開く", display_text="PDF")

display_order = ["★"] + [c for c in disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

st.data_editor(
    disp[display_order],
    key="main_editor",
    on_change=handle_main_editor_change,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    disabled=[c for c in display_order if c != "★"],  # ★のみ編集可
    height=520,
    num_rows="fixed",
)


# --- お気に入り一覧ヘッダー＋全外しボタン（横並び） ---
c1, c2 = st.columns([6, 1])
with c1:
    st.subheader(f"⭐ お気に入り（{len(st.session_state.favs)} 件）")
with c2:
    if st.button("❌ 全て外す", key="clear_favs_header", use_container_width=True):
        st.session_state.favs = set()
        st.rerun()

# お気に入り一覧（フィルタ無視で全体から）＋ tags 列（編集可）
visible_cols_full = make_visible_cols(df)

# ★ こちらも同じように summary を著者の右へ
if "著者" in visible_cols_full and "summary" in df.columns:
    idx = visible_cols_full.index("著者")
    if "summary" not in visible_cols_full:
        visible_cols_full.insert(idx + 1, "summary")

fav_disp_full = df.loc[:, visible_cols_full].copy()
fav_disp_full["_row_id"] = fav_disp_full.apply(make_row_id, axis=1)
fav_disp = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

def tags_str_for(rid: str) -> str:
    s = st.session_state.fav_tags.get(rid, set())
    return ", ".join(sorted(s)) if s else ""

if not fav_disp.empty:
    fav_disp["★"] = fav_disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)
    fav_disp["tags"] = fav_disp["_row_id"].apply(tags_str_for)  # ← 表示＆編集に使う

    fav_display_order = ["★"] + [c for c in fav_disp.columns if c not in ["★", "_row_id"]] + ["_row_id"]

    fav_column_config = {
        "★": st.column_config.CheckboxColumn("★", help="チェックで解除/追加", default=True, width="small"),
        "tags": st.column_config.TextColumn("tags（カンマ/空白区切り）", help="例: 清酒, 乳酸菌"),
    }
    if "HPリンク先" in fav_disp.columns:
        fav_column_config["HPリンク先"] = st.column_config.LinkColumn("HPリンク先", display_text="HP")
    if "PDFリンク先" in fav_disp.columns:
        fav_column_config["PDFリンク先"] = st.column_config.LinkColumn("PDFリンク先", display_text="PDF")

    st.data_editor(
        fav_disp[fav_display_order],
        key="fav_editor",
        on_change=update_fav_and_tags_from_favs,
        use_container_width=True,
        hide_index=True,
        column_config=fav_column_config,
        disabled=[c for c in fav_display_order if c not in ["★", "tags"]],
        height=420,
        num_rows="fixed"
    )
else:
    st.info("お気に入りは未選択です。上の表の『★』にチェックしてから反映してください。")

# -------------------- タグでお気に入りを絞り込み（AND/OR） --------------------
with st.expander("🔎 タグでお気に入りを絞り込み（AND/OR）", expanded=False):
    tag_query = st.text_input("タグ検索（カンマ/空白区切り）", key="tag_query")
    tag_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="tag_mode")

    fav_disp_for_filter = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()
    if tag_query.strip():
        tags = [t.strip() for t in re.split(r"[ ,，、；;　]+", tag_query) if t.strip()]
        def match_tags_row(row):
            row_tags = st.session_state.fav_tags.get(row["_row_id"], set())
            return all(t in row_tags for t in tags) if tag_mode == "AND" else any(t in row_tags for t in tags)
        fav_disp_for_filter = fav_disp_for_filter[fav_disp_for_filter.apply(match_tags_row, axis=1)]

    # 表示
    def tags_str_for_filter(rid: str) -> str:
        s = st.session_state.fav_tags.get(rid, set())
        return ", ".join(sorted(s)) if s else ""
    fav_disp_for_filter["tags"] = fav_disp_for_filter["_row_id"].apply(tags_str_for_filter)

    show_cols = ["No.","発行年","巻数","号数","論文タイトル","著者","対象物_top3","研究タイプ","HPリンク先","PDFリンク先","tags"]
    show_cols = [c for c in show_cols if c in fav_disp_for_filter.columns]
    st.dataframe(fav_disp_for_filter[show_cols], use_container_width=True, hide_index=True)

# -------------------- 下部アクション（CSV出力：2種類） --------------------
st.caption(
    f"現在のお気に入り：{len(st.session_state.favs)} 件 / "
    f"タグ数：{len({t for s in st.session_state.fav_tags.values() for t in s})} 種"
)

# 1) 絞り込み結果の出力（画面の検索結果テーブルと同じ列）
filtered_export_df = disp.drop(columns=["★", "_row_id"], errors="ignore")

# 2) お気に入りの出力（tags 列を付与）
fav_export = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

def _tags_join(rid: str) -> str:
    s = st.session_state.fav_tags.get(rid, set())
    return ", ".join(sorted(s)) if s else ""

fav_export["tags"] = fav_export["_row_id"].map(_tags_join)
fav_export = fav_export.drop(columns=["_row_id"], errors="ignore")

c_dl1, c_dl2 = st.columns(2)

with c_dl1:
    st.download_button(
        "📥 絞り込み結果をCSV出力（表示列のみ）",
        data=filtered_export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"filtered_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with c_dl2:
    st.download_button(
        "⭐ お気に入りをCSV出力（tags付き）",
        data=fav_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"favorites_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=fav_export.empty
    )