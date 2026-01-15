# -*- coding: utf-8 -*-
import re

def normalize_author_string(s: str) -> str:
    """全角/半角空白、読点の統一、セミコロン区切り正規化"""
    if not isinstance(s, str):
        return ""
    t = s.replace("、", ",").replace("，", ",")
    # 空白統一
    t = re.sub(r"[\u3000\s]+", " ", t).strip()
    # 区切りをセミコロンに
    t = t.replace(",", ";")
    # 余計なセミコロンの整理
    t = re.sub(r"\s*;\s*", "; ", t)
    return t

def normalize_query(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = t.replace("　", " ")
    return t
