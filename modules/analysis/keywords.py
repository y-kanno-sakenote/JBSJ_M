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

def _order_options(all_options: list[str], preferred: list[str]) -> list[str]:
    """preferred に含まれるものはその順で先頭に、それ以外は五十音（アルファベット）順で後ろに並べる"""
    s = set(all_options)
    head = [x for x in preferred if x in s]
    tail = sorted([x for x in all_options if x not in preferred])
    return head + tail

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

# 永続キャッシュIO（あれば使う）
try:
    from modules.common.cache_utils import cache_csv_path, load_csv_if_exists, save_csv
    HAS_DISK_CACHE = True
except Exception:
    HAS_DISK_CACHE = False


# ========= ユーティリティ =========
_SPLIT_MULTI_RE = re.compile(r"[;；,、，/／|｜\s　]+")

KEY_COLS = [
    "llm_keywords","primary_keywords","secondary_keywords","featured_keywords",
    "キーワード1","キーワード2","キーワード3","キーワード4","キーワード5",
    "キーワード6","キーワード7","キーワード8","キーワード9","キーワード10",
]

# ---- ストップワード（英語＋日本語の汎用ノイズ + 'nan'）----
STOPWORDS = set([
    # 英語系
    "and","the","of","to","in","for","on","at","with","by","an","is","are",
    "this","that","it","as","be","from","was","were","or","a","we","our",
    "their","can","may","will","using","use","used","study","based",
    "analysis","data","result","results","method","methods","conclusion",
    "discussion","introduction","materials","material","supplementary",
    "figure","table","et","al","etc","between","among","within","into",
    "over","under","than","then","there","here","such","these","those",
    "however","therefore","thus","because","due","per","based","according",
    "observed","obtained","present","presented","approach","paper","research",
    "nan","none","null",
    # 日本語系（助詞・形式名詞・汎用ノイズ）
    "これ","それ","あれ","ため","もの","こと","よう","また","および","およびび",
    "における","について","により","による","など","する","した","して","され","される",
    "いる","ある","なる","できる","可能","結果","方法","目的","考察","結論","序論",
    "図","表","例","例えば","本研究","本論文","本報","本報告","本稿","一方","一方で",
    "さらに","しかし","そこで","まず","次に","最後","以上","以下","本","各","本学","同",
])

def norm_key(s: str) -> str:
    s = str(s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def split_multi(s) -> List[str]:
    if not s:
        return []
    return [w.strip() for w in _SPLIT_MULTI_RE.split(str(s)) if w.strip()]

def col_contains_any(df_col: pd.Series, needles: List[str]) -> pd.Series:
    if not needles:
        return pd.Series([True] * len(df_col), index=df_col.index)
    lo_needles = [norm_key(n) for n in needles]
    def _hit(v: str) -> bool:
        s = norm_key(v)
        return any(n in s for n in lo_needles)
    return df_col.fillna("").astype(str).map(_hit)

@st.cache_data(ttl=600, show_spinner=False)
def year_min_max(df: pd.DataFrame) -> Tuple[int, int]:
    if "発行年" not in df.columns:
        return (1980, 2025)
    y = pd.to_numeric(df["発行年"], errors="coerce")
    if y.notna().any():
        return (int(y.min()), int(y.max()))
    return (1980, 2025)

def _apply_filters(df: pd.DataFrame,
                   y_from: int, y_to: int,
                   targets: List[str], types: List[str]) -> pd.DataFrame:
    use = df.copy()
    if "発行年" in use.columns:
        y = pd.to_numeric(use["発行年"], errors="coerce")
        use = use[(y >= y_from) & (y <= y_to) | y.isna()]
    if targets and "対象物_top3" in use.columns:
        use = use[col_contains_any(use["対象物_top3"], targets)]
    if types and "研究タイプ_top3" in use.columns:
        use = use[col_contains_any(use["研究タイプ_top3"], types)]
    return use

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

def _draw_pyvis_from_edges(edges: pd.DataFrame, height_px: int = 650) -> None:
    if not (HAS_NX and HAS_PYVIS):
        st.info("networkx / pyvis が未導入のため、表のみ表示しています。")
        return
    if edges.empty:
        st.warning("対象条件でエッジがありません。")
        return

    # 文字列IDに統一
    G = nx.Graph()
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"]); w = int(r["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(gravity=-30000, central_gravity=0.25, spring_length=120, spring_strength=0.02)
    net.from_nx(G)
    html = net.generate_html(notebook=False)  # ← ブラウザ自動オープン回避
    st.components.v1.html(html, height=height_px, scrolling=True)

# ==== 追加：安全表示ヘルパー（UIは変えずに落ちにくく） ====
def safe_show_image(obj: Any) -> None:
    """
    Streamlitの画像表示で型差異があっても落ちないように安全に表示する。
    UI（描画結果）は変更しない。
    """
    import numpy as np
    import io
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore

    # None
    if obj is None:
        st.warning("画像データが None でした。生成に失敗している可能性があります。")
        return

    # Matplotlib Figure -> pyplot
    try:
        import matplotlib.figure
        if isinstance(obj, matplotlib.figure.Figure):
            st.pyplot(obj)
            return
    except Exception:
        pass

    # PIL.Image は必ず PNG バイト列へ変換してから表示（環境差対策）
    if Image is not None and isinstance(obj, Image.Image):
        try:
            img = obj
            # 透過やパレット等のモードを統一
            if img.mode not in ("RGB", "RGBA"):
                # パレットやF/LAなどはRGBA化が安全
                img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")  # フォーマットはPNG固定で安定
            st.image(buf.getvalue(), use_container_width=True)
        except Exception as e:
            st.warning(f"PIL画像の表示で例外が発生しました: {e!s}")
        return

    # NumPy array
    if isinstance(obj, np.ndarray):
        arr = obj
        # 形状チェック
        if arr.ndim == 2:
            pass  # gray OK
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass
        else:
            st.warning(f"想定外の配列shapeです: {arr.shape}")
            return
        # dtypeをuint8へ
        if arr.dtype in (np.float32, np.float64):
            a = arr
            if np.nanmax(a) <= 1.0:
                a = (np.nan_to_num(a) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                a = np.nan_to_num(a).clip(0, 255).astype(np.uint8)
            st.image(a, use_container_width=True)
        elif arr.dtype == np.uint8:
            st.image(arr, use_container_width=True)
        else:
            a = np.nan_to_num(arr).clip(0, 255).astype(np.uint8)
            st.image(a, use_container_width=True)
        return

    # bytes / bytearray
    if isinstance(obj, (bytes, bytearray)):
        st.image(obj, use_container_width=True)
        return

    # 文字列（URL or パス）
    if isinstance(obj, str):
        st.image(obj, use_container_width=True)
        return

    # それ以外
    st.warning(f"st.imageが扱えない型でした: {type(obj)}")
    
# ========= ① 頻出キーワード =========
def _render_freq_block(df: pd.DataFrame) -> None:
    st.markdown("### ① 頻出キーワード")

    ymin, ymax = year_min_max(df)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_freq_year")

    # ▼ 候補リストを自動抽出
    targets_all = sorted({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                          for w in _SPLIT_MULTI_RE.split(v) if w.strip()})
    types_all   = sorted({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                          for w in _SPLIT_MULTI_RE.split(v) if w.strip()})

    # ★ 表示順を固定
    targets_all = _order_options(targets_all, TARGET_ORDER)
    types_all   = _order_options(types_all, TYPE_ORDER)

    with c2:
        tg_needles = st.multiselect("対象物で絞り込み", options=targets_all, default=[], key="kw_freq_tg")
    with c3:
        tp_needles = st.multiselect("研究タイプで絞り込み", options=types_all, default=[], key="kw_freq_tp")
    with c4:
        topn = st.number_input("表示件数", min_value=5, max_value=100, value=30, step=5, key="kw_freq_topn")

    # ▼ フィルタ反映
    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)
    freq = keyword_freq(use)
    freq_df = _freq_to_df(freq, int(topn))

    if freq_df.empty:
        st.info("条件に合うキーワードが見つかりませんでした。")
        return

    # グラフ
    if HAS_PX:
        fig = px.bar(freq_df, x="キーワード", y="件数", text_auto=True, title="頻出キーワード（上位）")
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(freq_df.set_index("キーワード")["件数"])

    # WordCloud（任意・ボタン生成）
    with st.expander("☁ WordCloud（任意）", expanded=False):
        if HAS_WC:
            if st.button("生成する", key="kw_wc_btn"):
                textfreq = {row["キーワード"]: int(row["件数"]) for _, row in freq_df.iterrows()}
                # 日本語フォント対応（見つかれば適用）
                font_path = _get_japanese_font_path()
                wc = WordCloud(width=900, height=450, background_color="white",
                               collocations=False, prefer_horizontal=1.0,
                               font_path=font_path or None)
                img = wc.generate_from_frequencies(textfreq).to_image()
                # --- ここだけ差し替え（UI変更なし） ---
                safe_show_image(img)
        else:
            st.caption("※ wordcloud が未導入のため非表示です。")

# ========= ② 共起ネットワーク（遅延描画） =========
def _render_cooccur_block(df: pd.DataFrame) -> None:
    st.markdown("### ② 共起キーワードネットワーク")

    ymin, ymax = year_min_max(df)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_co_year")
    with c2:
        min_edge = st.number_input("エッジ最小回数 (w≥)", min_value=1, max_value=50, value=3, step=1, key="kw_co_minw")
    with c3:
        topN = st.number_input("ノード上限（出現上位）", min_value=30, max_value=300, value=120, step=10, key="kw_co_topn")
    with c4:
        st.caption("下のボタンで描画します。")

    # ▼ 候補リストを自動抽出
    targets_all = sorted({w for v in df.get("対象物_top3", pd.Series(dtype=str)).fillna("")
                          for w in _SPLIT_MULTI_RE.split(v) if w.strip()})
    types_all   = sorted({w for v in df.get("研究タイプ_top3", pd.Series(dtype=str)).fillna("")
                          for w in _SPLIT_MULTI_RE.split(v) if w.strip()})

    # ★ 表示順を固定
    targets_all = _order_options(targets_all, TARGET_ORDER)
    types_all   = _order_options(types_all, TYPE_ORDER)

    c5, c6 = st.columns([1,1])
    with c5:
        tg_needles = st.multiselect("対象物で絞り込み", options=targets_all, default=[], key="kw_co_tg")
    with c6:
        tp_needles = st.multiselect("研究タイプで絞り込み", options=types_all, default=[], key="kw_co_tp")

    # ▼ フィルタ反映
    use = _apply_filters(df, y_from, y_to, tg_needles, tp_needles)

    # --- キャッシュと描画ロジックはそのまま ---
    cache_key = f"kwco|{y_from}-{y_to}|min{min_edge}|top{topN}|tg{','.join(tg_needles)}|tp{','.join(tp_needles)}"
    edges = build_keyword_cooccur_edges(use, int(min_edge))
    if not edges.empty and int(topN) > 0:
        deg = pd.concat([edges.groupby("src")["weight"].sum(),
                         edges.groupby("dst")["weight"].sum()], axis=1).fillna(0).sum(axis=1)
        keep_nodes = set(deg.sort_values(ascending=False).head(int(topN)).index.tolist())
        edges = edges[edges["src"].isin(keep_nodes) & edges["dst"].isin(keep_nodes)].reset_index(drop=True)

    st.caption(f"エッジ数: {len(edges)}")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)

    with st.expander("🕸️ ネットワークを描画（PyVis / 任意依存）", expanded=False):
        if HAS_PYVIS and HAS_NX:
            if st.button("🌐 描画する", key="kw_co_draw"):
                _draw_pyvis_from_edges(edges, height_px=680)
        else:
            st.info("networkx / pyvis が未導入のため、表のみ表示しています。")
            
# ========= ③ トレンド（経年変化） =========
def _render_trend_block(df: pd.DataFrame) -> None:
    st.markdown("### ③ トレンド分析（経年変化）")

    ymin, ymax = year_min_max(df)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        y_from, y_to = st.slider("対象年（範囲）", min_value=ymin, max_value=ymax,
                                 value=(ymin, ymax), key="kw_trend_year")
    with c2:
        topn = st.number_input("表示する語数（TopN）", min_value=5, max_value=50, value=15, step=5, key="kw_trend_topn")
    with c3:
        ma = st.number_input("移動平均（年）", min_value=1, max_value=7, value=1, step=1, key="kw_trend_ma")

    use = _apply_filters(df, y_from, y_to, [], [])
    yearly = yearly_keyword_counts(use)
    if yearly.empty:
        st.info("データがありません。")
        return

    # 最新年付近のTopN語を選ぶ（全体上位だと凡例が多すぎるため）
    latest_year = yearly["発行年"].max()
    latest_top = (yearly[yearly["発行年"] == latest_year]
                  .sort_values("count", ascending=False)["keyword"]
                  .head(int(topn)).tolist())
    piv = (yearly[yearly["keyword"].isin(latest_top)]
           .pivot_table(index="発行年", columns="keyword", values="count", aggfunc="sum")
           .fillna(0).sort_index())

    if int(ma) > 1:
        piv = piv.rolling(window=int(ma), min_periods=1).mean()

    if HAS_PX:
        fig = px.line(piv.reset_index().melt(id_vars="発行年", var_name="キーワード", value_name="件数"),
                      x="発行年", y="件数", color="キーワード", markers=True)
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(piv)

# ========= エクスポート：タブ本体 =========
def render_keyword_tab(df: pd.DataFrame) -> None:
    st.markdown("## 🧠 キーワード分析")

    tab1, tab2, tab3 = st.tabs([
        "① 頻出キーワード",
        "② 共起ネットワーク",
        "③ トレンド分析",
    ])

    with tab1:
        _render_freq_block(df)

    with tab2:
        _render_cooccur_block(df)   # ← 遅延描画（ボタン式）

    with tab3:
        _render_trend_block(df)