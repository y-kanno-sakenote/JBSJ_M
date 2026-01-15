from __future__ import annotations
from typing import Iterable, List

def normalize_authors(raw: str) -> List[str]:
    """著者名の全角/半角・空白の揺らぎを軽く正規化して分割する簡易関数。"""
    s = raw.replace("，", ",").replace("、", ",").replace("　", " ").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts
