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

import re, time
import pandas as pd
import streamlit as st
from pathlib import Path
from modules.analysis import render_analysis_tab

# ---- try to reuse common filter constants/helpers (no UI/logic change) ----
try:
    from modules.common.filters import (
        TARGET_ORDER as _COMMON_TARGET_ORDER,
        TYPE_ORDER as _COMMON_TYPE_ORDER,
    )
    _HAS_COMMON_FILTERS = True
except Exception:
    _COMMON_TARGET_ORDER = None
    _COMMON_TYPE_ORDER = None
    _HAS_COMMON_FILTERS = False

# optional helper import (best-effort)
try:
    from modules.common.filters import sort_with_order as _common_sort_with_order  # may not exist
except Exception:
    _common_sort_with_order = None
import os

# 受け取り: クエリパラメータから著者を事前反映
qp = st.query_params
prefill_author = qp.get("author", None)


# -------------------- ページ設定 --------------------
st.set_page_config(page_title="醸造協会誌 論文検索", layout="wide")

# -------------------- simple auth (local) --------------------
try:
    from modules.common import auth
except Exception:
    auth = None

def _ensure_session_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

_ensure_session_auth()

# Safe rerun: some Streamlit environments may not allow experimental_rerun()
def _safe_rerun():
    try:
        # experimental_rerun may raise if Streamlit internals change or when
        # called outside a normal session. Guard it to avoid crashing the app.
        st.experimental_rerun()
    except Exception:
        # best-effort: set a flag so UI can reflect the change without raising
        try:
            st.session_state._needs_rerun = True
        except Exception:
            pass

with st.sidebar:
    st.subheader("Login")
    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            _safe_rerun()
        # --- admin panel: manage users/plans ---
        try:
            if st.session_state.username == "admin":
                st.markdown("---")
                st.subheader("管理パネル")
                try:
                    users = auth.load_users()
                except Exception:
                    users = {}

                # display simple user list with current plan shown
                user_list = sorted(users.keys())
                # build display labels like: "alice — free"
                display_map = {}
                display_opts = ["(選択してください)"]
                for u in user_list:
                    plan = users.get(u, {}).get("plan", "free")
                    label = f"{u} — {plan}"
                    display_map[label] = u
                    display_opts.append(label)

                sel_label = st.selectbox("ユーザー選択", display_opts, key="_admin_sel_user")
                if sel_label and sel_label != "(選択してください)":
                    sel_user = display_map.get(sel_label)
                    cur_plan = auth.get_plan(sel_user)
                    st.write(f"現在の plan: {cur_plan}")
                    col_a, col_b = st.columns([1,1])
                    with col_a:
                        if st.button("Set free", key=f"set_free_{sel_user}"):
                            if auth.set_plan(sel_user, "free"):
                                st.success(f"{sel_user} を free に設定しました")
                                _safe_rerun()
                            else:
                                st.error("更新に失敗しました")
                    with col_b:
                        if st.button("Set paid", key=f"set_paid_{sel_user}"):
                            if auth.set_plan(sel_user, "paid"):
                                st.success(f"{sel_user} を paid に設定しました")
                                _safe_rerun()
                            else:
                                st.error("更新に失敗しました")

                st.markdown("**新規ユーザー追加**")
                new_user = st.text_input("Username", key="_admin_new_user")
                new_pwd = st.text_input("Password", type="password", key="_admin_new_pwd")
                new_plan = st.selectbox("Plan", ["free","paid"], index=0, key="_admin_new_plan")
                if st.button("Add user", key="_admin_add_user"):
                    if not new_user or not new_pwd:
                        st.error("ユーザー名とパスワードを入力してください")
                    else:
                        ok = auth.add_user(new_user, new_pwd)
                        if ok:
                            # ensure plan set
                            auth.set_plan(new_user, new_plan)
                            st.success(f"ユーザー {new_user} を追加しました（plan={new_plan}）")
                            _safe_rerun()
                        else:
                            st.error("ユーザー追加に失敗しました")
        except Exception:
            # guard against auth errors in environments without auth
            pass
    else:
        user = st.text_input("Username", key="_login_user")
        pwd = st.text_input("Password", type="password", key="_login_pwd")
        if st.button("Login"):
            ok = False
            if auth is not None:
                try:
                    ok = auth.verify_user(user, pwd)
                except Exception:
                    ok = False
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.success("Login successful")
                _safe_rerun()
            else:
                st.error("Login failed")

if not st.session_state.logged_in:
    st.stop()

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
      max-height: 70vh !important;
      min-height: 360px !important;
      overflow-y: auto !important;
      padding-right: 6px !important;
      scrollbar-width: auto;
      scrollbar-color: #555 #e9e9e9;
    }
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

    /* WebKit 系（Chrome/Edge/Safari）のスクロールバー */
    ul[role="listbox"]::-webkit-scrollbar { width: 16px; }
    ul[role="listbox"]::-webkit-scrollbar-track { background: #e9e9e9; border-radius: 8px; }
    ul[role="listbox"]::-webkit-scrollbar-thumb { background: #555; border-radius: 8px; border: 3px solid #e9e9e9; }
    ul[role="listbox"]::-webkit-scrollbar-thumb:hover { background: #333; }
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
# 統一定義（存在すれば common/filters の順序を使用）
TARGET_ORDER = _COMMON_TARGET_ORDER or [
    "清酒","ビール","ワイン","焼酎","アルコール飲料","発酵乳・乳製品",
    "醤油","味噌","発酵食品","農産物・果実","副産物・バイオマス","酵母・微生物","アミノ酸・タンパク質","その他"
]
TYPE_ORDER = _COMMON_TYPE_ORDER or [
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
    """
    1) テンプレの順 2) 未収載はアルファ順 3) その他は最後
    ※ modules.common.filters に sort_with_order があればそれを優先使用（見た目の順序は同じ）
    """
    # best-effort: use shared sorter if present (keeps UI order identical)
    if _common_sort_with_order is not None:
        try:
            return _common_sort_with_order(list(dict.fromkeys(values)), template)
        except Exception:
            pass
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
st.title("醸造協会誌　論文検索 β_3.0")

DEMO_CSV_PATH = Path("data/keywords_summary5.csv")   # メインCSV
SUMMARY_CSV_PATH = Path("data/summaries.csv")         # ← 追加: summary
AUTHORS_CSV_PATH = Path("data/authors_readings.csv")  # ← 追加: 著者読み


@st.cache_data(ttl=600, show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return ensure_cols(pd.read_csv(path, encoding="utf-8"))


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
        adf = adf.drop_duplicates(subset=["reading"], keep="first")
        return adf
    except Exception:
        return None

# -------------------- データ読み込み（固定：同梱CSV） --------------------
df = None
if DEMO_CSV_PATH.exists():
    df = load_local_csv(DEMO_CSV_PATH)
    st.caption(f"✅ デモCSVを読み込みました: {DEMO_CSV_PATH}")
else:
    st.error(f"データファイルが見つかりません: {DEMO_CSV_PATH}")
    st.stop()
# --- summary をマージ ---
sum_df = load_summaries(SUMMARY_CSV_PATH)
if sum_df is not None:
    df = df.merge(sum_df, on="file_name", how="left")

 # ===================== タブ切り替え =====================
tab_search, tab_analysis = st.tabs(["🔍 検索", "📊 分析"])

# 永続キャッシュは本番では常時ON（UIは廃止）
use_disk_cache = True

with tab_search:
    # -------------------- 検索フィルタ（分析タブと順序を揃える） --------------------
    # 1段目：発行年・対象物・研究タイプ
    st.subheader("検索フィルタ")
    year_vals = pd.to_numeric(df.get("発行年", pd.Series(dtype=str)), errors="coerce")
    if year_vals.notna().any():
        ymin_all, ymax_all = int(year_vals.min()), int(year_vals.max())
    else:
        ymin_all, ymax_all = 1980, 2025

    row1_y, row1_tg, row1_tp = st.columns([1.0, 1.2, 1.2])
    with row1_y:
        y_from, y_to = st.slider(
            "発行年（範囲）", min_value=ymin_all, max_value=ymax_all,
            value=(ymin_all, ymax_all)
        )
    with row1_tg:
        raw_targets = {t for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        targets_all = order_by_template(list(raw_targets), TARGET_ORDER)
        targets_sel = st.multiselect("対象物で絞り込み", targets_all, default=[])
    with row1_tp:
        raw_types = {t for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("") for t in split_multi(v)}
        types_all = order_by_template(list(raw_types), TYPE_ORDER)
        types_sel = st.multiselect("研究タイプで絞り込み", types_all, default=[])

    # 2段目：著者・著者イニシャル選択
    col_author, col_initial = st.columns([1.5, 2])
    with col_initial:
        initials = ["すべて","あ","か","さ","た","な","は","ま","や","ら","わ","英字"]
        if "author_initial" not in st.session_state:
            st.session_state.author_initial = "すべて"
        st.radio(
            "著者イニシャル選択",
            options=initials,
            horizontal=True,
            key="author_initial",
        )
    ini = st.session_state["author_initial"]
    with col_author:
        adf = load_authors_readings(AUTHORS_CSV_PATH)
        if adf is not None and not adf.empty:
            cand = adf.copy()
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
                s = str(s or "");  return kata_to_hira(s)[0] if s else None
            def is_roman_head(s: str) -> bool:
                return bool(re.match(r"[A-Za-z]", str(s or "")))

            if ini == "英字":
                cand = cand[cand["reading"].astype(str).str.match(r"[A-Za-z]")]
            elif ini != "すべて":
                allowed = set(GOJUON.get(ini, ""))
                cand = cand[cand["reading"].apply(
                    lambda s: (not is_roman_head(s)) and (hira_head(s) in allowed if hira_head(s) else False)
                )]

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
            authors_sel_readings = st.multiselect(
                "著者（読みで検索可 / 表示は漢字＋読み）",
                options=options_readings,
                format_func=lambda r: f"{reading2author.get(r, r)}｜{r}",
                placeholder="例：やまだ / さとう / たかはし ..."
            )
            authors_sel = sorted({reading2author[r] for r in authors_sel_readings}) if authors_sel_readings else []
        else:
            authors_all = build_author_candidates(df)
            authors_sel = st.multiselect("著者", authors_all, default=[])

    if 'authors_sel' not in locals(): authors_sel = []
    if 'targets_sel' not in locals(): targets_sel = []
    if 'types_sel'   not in locals(): types_sel   = []

    # 3段目：キーワード
    kw_row1, kw_row2 = st.columns([3, 1])
    with kw_row1:
        # keep kw_query in session to allow other UI elements to update it safely
        if "kw_query_input" not in st.session_state:
            st.session_state["kw_query_input"] = ""
        # bind the text_input to a stable session-state key (avoids modifying session_state after widget init)
        st.text_input("キーワード（空白/カンマで複数可）", value=st.session_state.get("kw_query_input", ""), key="kw_query_input")
        kw_query = st.session_state.get("kw_query_input", "")
    with kw_row2:
        kw_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="kw_mode")

    # -------------------- フィルタ適用 --------------------
    def apply_filters(_df: pd.DataFrame) -> pd.DataFrame:
        df2 = _df.copy()
        if "発行年" in df2.columns:
            y = pd.to_numeric(df2["発行年"], errors="coerce")
            df2 = df2[(y >= y_from) & (y <= y_to) | y.isna()]
        if authors_sel and "著者" in df2.columns:
            sel = {norm_key(a) for a in authors_sel}
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

    # -------------------- 出典・再現性用の一行バナー（検索タブ用） --------------------
    def _render_provenance_banner(total_n: int,
                                  filtered_n: int,
                                  y_from: int, y_to: int,
                                  authors: list[str] | None = None,
                                  targets: list[str] | None = None,
                                  types: list[str] | None = None,
                                  kw_query: str | None = None,
                                  kw_mode: str = "OR") -> None:
        """
        タブ上部に軽量の共通注記（出典＋適用条件）を1行で表示。
        空の要素は自動的に省略。
        """
        parts = [f"出典：JBSJ DB（N={filtered_n} / {total_n}）", f"期間：{y_from}–{y_to}"]

        def _fmt_list(name: str, vals: list[str] | None, max_items: int = 6) -> None:
            if not vals:
                return
            vs = [str(v) for v in vals if str(v).strip()]
            if not vs:
                return
            txt = ", ".join(vs[:max_items]) + (" …" if len(vs) > max_items else "")
            parts.append(f"{name}：{txt}")

        _fmt_list("対象物", targets)
        _fmt_list("研究タイプ", types)
        _fmt_list("著者", authors)

        if kw_query and kw_query.strip():
            q = "、".join([t for t in re.split(r"[ ,，、；;　]+", kw_query.strip()) if t])
            if q:
                parts.append(f"キーワード（{kw_mode}）：{q}")

        line = " ｜ ".join(parts)

        st.caption(
            f"{line}",
        )

    _render_provenance_banner(
        total_n=len(df),
        filtered_n=len(filtered),
        y_from=y_from, y_to=y_to,
        authors=authors_sel or [],
        targets=targets_sel or [],
        types=types_sel or [],
        kw_query=kw_query,
        kw_mode=kw_mode
    )

#    # -------------------- 検索結果テーブル --------------------
    st.markdown(f"### 検索結果（{len(filtered):,}件）")

    visible_cols = make_visible_cols(filtered)
    if "著者" in visible_cols and "summary" in filtered.columns:
        idx = visible_cols.index("著者")
        if "summary" not in visible_cols:
            visible_cols.insert(idx + 1, "summary")

    disp = filtered.loc[:, visible_cols].copy()
    # Ensure 終了ページ is available for display even if make_visible_cols hid it
    if "終了ページ" in filtered.columns and "終了ページ" not in disp.columns:
        disp["終了ページ"] = filtered["終了ページ"]
    disp["_row_id"] = disp.apply(make_row_id, axis=1)

    # 表示用に発行年のカンマ(,)を除去（例: "1,988" -> "1988"）
    if "発行年" in disp.columns:
        try:
            disp["発行年"] = disp["発行年"].astype(str).str.replace(",", "").str.strip()
        except Exception:
            pass

    if "favs" not in st.session_state:
        st.session_state.favs = set()
    if "fav_tags" not in st.session_state:
        st.session_state.fav_tags = {}

    disp["★"] = disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)

    # 列名修正（開始ページ → p.始 / 終了ページ → p.終）
    rename_map = {}
    if "開始ページ" in disp.columns:
        rename_map["開始ページ"] = "p.始"
    if "終了ページ" in disp.columns:
        rename_map["終了ページ"] = "p.終"
    if rename_map:
        disp = disp.rename(columns=rename_map)

    # LinkColumn 設定（HP/PDF を短い見出しで）
    column_config = {
        "★": st.column_config.CheckboxColumn("★", help="気になる論文にチェック/解除", default=False, width="small"),
    }
    if "HPリンク先" in disp.columns:
        column_config["HPリンク先"] = st.column_config.LinkColumn("HP", help="外部サイトへ移動", display_text="HP")
    if "PDFリンク先" in disp.columns:
        column_config["PDFリンク先"] = st.column_config.LinkColumn("PDF", help="PDFを開く", display_text="PDF")

    # 列順：スクリーンショット順に合わせる
    desired_front = ["★", "No.", "HPリンク先", "PDFリンク先", "発行年", "巻数", "号数", "p.始", "p.終"]
    # include only those that exist in disp
    front = [c for c in desired_front if c in disp.columns]
    rest = [c for c in disp.columns if c not in set(front) and c != "_row_id"]
    display_order = front + rest + ["_row_id"]

    with st.form("main_table_form", clear_on_submit=False):
        # (per-row immediate toggle removed — keep session-only checkboxes in the table)
        edited_main = st.data_editor(
            disp[display_order],
            key="main_editor",
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            disabled=[c for c in display_order if c != "★"],
            height=520,
            num_rows="fixed",
        )
        apply_main = st.form_submit_button("チェックした論文をお気に入りリストに追加", use_container_width=True)

    if apply_main:
        subset_ids_main = set(disp["_row_id"].tolist())
        checked_subset_main = set(edited_main.loc[edited_main["★"] == True, "_row_id"].tolist())
        st.session_state.favs = (st.session_state.favs - subset_ids_main) | checked_subset_main

    # --- お気に入り一覧ヘッダー＋全外しボタン ---
    c1, c2 = st.columns([6, 1])
    with c1:
        st.subheader(f"⭐ お気に入り（{len(st.session_state.favs)} 件）")
    with c2:
        if st.button("❌ 全て外す", key="clear_favs_header", use_container_width=True):
            st.session_state.favs = set()
            st.rerun()

    # お気に入り一覧（フィルタ無視で全体から）＋ tags 列（編集可）
    visible_cols_full = make_visible_cols(df)
    if "著者" in visible_cols_full and "summary" in df.columns:
        idx = visible_cols_full.index("著者")
        if "summary" not in visible_cols_full:
            visible_cols_full.insert(idx + 1, "summary")

    fav_disp_full = df.loc[:, visible_cols_full].copy()
    fav_disp_full["_row_id"] = fav_disp_full.apply(make_row_id, axis=1)
    fav_disp = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()

    # お気に入り表示用にも発行年のカンマを除去
    if "発行年" in fav_disp_full.columns:
        try:
            fav_disp_full["発行年"] = fav_disp_full["発行年"].astype(str).str.replace(",", "").str.strip()
        except Exception:
            pass

    def tags_str_for(rid: str) -> str:
        s = st.session_state.fav_tags.get(rid, set())
        return ", ".join(sorted(s)) if s else ""

    if not fav_disp.empty:
        # 列名修正（開始ページ → p.始 / 終了ページ → p.終）
        rename_map = {}
        if "開始ページ" in fav_disp.columns:
            rename_map["開始ページ"] = "p.始"
        if "終了ページ" in fav_disp.columns:
            rename_map["終了ページ"] = "p.終"
        if rename_map:
            fav_disp = fav_disp.rename(columns=rename_map)

        fav_disp["★"] = fav_disp["_row_id"].apply(lambda rid: rid in st.session_state.favs)
        fav_disp["tags"] = fav_disp["_row_id"].apply(tags_str_for)

        fav_column_config = {
            "★": st.column_config.CheckboxColumn("★", help="チェックで解除/追加（下のボタンで反映）", default=True, width="small"),
            "tags": st.column_config.TextColumn("tags（カンマ/空白区切り）", help="例: 清酒, 乳酸菌"),
        }
        if "HPリンク先" in fav_disp.columns:
            fav_column_config["HPリンク先"] = st.column_config.LinkColumn("HP", display_text="HP")
        if "PDFリンク先" in fav_disp.columns:
            fav_column_config["PDFリンク先"] = st.column_config.LinkColumn("PDF", display_text="PDF")

        # 列順調整：スクリーンショット順に合わせる
        desired_front = ["★", "No.", "HPリンク先", "PDFリンク先", "発行年", "巻数", "号数", "p.始", "p.終"]
        front = [c for c in desired_front if c in fav_disp.columns]
        rest = [c for c in fav_disp.columns if c not in set(front) and c not in {"_row_id", "tags"}]
        fav_display_order = front + rest + ["tags", "_row_id"]

        with st.form("fav_table_form", clear_on_submit=False):
            fav_edited = st.data_editor(
                fav_disp[fav_display_order],
                key="fav_editor",
                use_container_width=True,
                hide_index=True,
                column_config=fav_column_config,
                disabled=[c for c in fav_display_order if c not in ["★", "tags"]],
                height=420,
                num_rows="fixed",
            )
            apply_fav = st.form_submit_button("お気に入りの変更（★/tags）を更新", use_container_width=True)

        if apply_fav:
            subset_ids_fav = set(fav_disp["_row_id"].tolist())
            fav_checked_subset = set(fav_edited.loc[fav_edited["★"] == True, "_row_id"].tolist())
            st.session_state.favs = (st.session_state.favs - subset_ids_fav) | fav_checked_subset

            def parse_tags(s):
                if not isinstance(s, str): s = str(s or "")
                parts = [t.strip() for t in re.split(r"[ ,，、；;　]+", s) if t.strip()]
                return set(parts)
            for _, r in fav_edited.iterrows():
                rid = r["_row_id"]
                tag_set = parse_tags(r.get("tags", ""))
                if tag_set:
                    st.session_state.fav_tags[rid] = tag_set
                elif rid in st.session_state.fav_tags:
                    del st.session_state.fav_tags[rid]

            st.success("お気に入り（★/tags）を反映しました")
            st.rerun()
    else:
        st.info("お気に入りは未選択です。上の表の『★』にチェックしてから反映してください。")
        fav_edited = None

    # -------------------- タグでお気に入りを絞り込み（AND/OR） --------------------
    with st.expander("🔎 タグでお気に入りを絞り込み（AND/OR）", expanded=False):
        tag_query = st.text_input("タグ検索（空白/カンマで複数可）", key="tag_query")
        tag_mode = st.radio("一致条件", ["OR", "AND"], index=0, horizontal=True, key="tag_mode")

        fav_disp_for_filter = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()
        if tag_query.strip():
            tags = [t.strip() for t in re.split(r"[ ,，、；;　]+", tag_query) if t.strip()]
            def match_tags_row(row):
                row_tags = st.session_state.fav_tags.get(row["_row_id"], set())
                return all(t in row_tags for t in tags) if tag_mode == "AND" else any(t in row_tags for t in tags)
            fav_disp_for_filter = fav_disp_for_filter[fav_disp_for_filter.apply(match_tags_row, axis=1)]

        def tags_str_for_filter(rid: str) -> str:
            s = st.session_state.fav_tags.get(rid, set())
            return ", ".join(sorted(s)) if s else ""
        fav_disp_for_filter["tags"] = fav_disp_for_filter["_row_id"].apply(tags_str_for_filter)

        show_cols = ["No.","発行年","巻数","号数","論文タイトル","著者","対象物_top3","研究タイプ","HPリンク先","PDFリンク先","tags"]
        show_cols = [c for c in show_cols if c in fav_disp_for_filter.columns]
        # ensure displayed year has no comma
        if "発行年" in fav_disp_for_filter.columns:
            try:
                fav_disp_for_filter["発行年"] = fav_disp_for_filter["発行年"].astype(str).str.replace(",", "").str.strip()
            except Exception:
                pass
        st.dataframe(fav_disp_for_filter[show_cols], use_container_width=True, hide_index=True)

    # -------------------- 下部アクション（CSV出力：2種類） --------------------
    st.caption(
        f"現在のお気に入り：{len(st.session_state.favs)} 件 / "
        f"タグ数：{len({t for s in st.session_state.fav_tags.values() for t in s})} 種"
    )

    filtered_export_df = disp.drop(columns=["★", "_row_id"], errors="ignore")
    if "発行年" in filtered_export_df.columns:
        try:
            filtered_export_df["発行年"] = filtered_export_df["発行年"].astype(str).str.replace(",", "").str.strip()
        except Exception:
            pass

    fav_export = fav_disp_full[fav_disp_full["_row_id"].isin(st.session_state.favs)].copy()
    def _tags_join(rid: str) -> str:
        s = st.session_state.fav_tags.get(rid, set())
        return ", ".join(sorted(s)) if s else ""
    fav_export["tags"] = fav_export["_row_id"].map(_tags_join)
    fav_export = fav_export.drop(columns=["_row_id"], errors="ignore")
    if "発行年" in fav_export.columns:
        try:
            fav_export["発行年"] = fav_export["発行年"].astype(str).str.replace(",", "").str.strip()
        except Exception:
            pass

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

with tab_analysis:
    render_analysis_tab(df, use_disk_cache=use_disk_cache)