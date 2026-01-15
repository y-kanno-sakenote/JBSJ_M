from __future__ import annotations
import re
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
STOPWORDS_JA = {"こと","もの","ため","など","よう","場合","および","及び","また","これ","それ","この","その","図","表","第","同","一方","または","又は","における","について","に対する"}
STOPWORDS_ALL = _WC | {s.casefold() for s in STOPWORDS_EN_EXTRA} | STOPWORDS_JA

_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
_NUM_RE        = re.compile(r"^\d+(\.\d+)?$")
_EN_SHORT_RE   = re.compile(r"^[A-Za-z]{1,2}$")

def clean_token(tok: str) -> str:
    if tok is None: return ""
    t = str(tok).strip()
    if not t: return ""
    t = _PUNCT_EDGE_RE.sub("", t)
    if not t: return ""
    low = t.casefold()
    if low in {"none", "nan"}: return ""
    if _NUM_RE.fullmatch(t): return ""
    if _EN_SHORT_RE.fullmatch(t): return ""
    if low in STOPWORDS_ALL: return ""
    return t