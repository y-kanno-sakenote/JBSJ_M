# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import Tuple
from config import DATA_PATH

def load_dataset(path: str | None = None) -> pd.DataFrame:
    """CSV/TSV/JSONいずれかを読み込む簡易ローダー。
    必要に応じて差し替え可。
    """
    p = path or DATA_PATH
    if p.endswith(".csv"):
        df = pd.read_csv(p)
    elif p.endswith(".tsv"):
        df = pd.read_csv(p, sep="\t")
    elif p.endswith(".json"):
        df = pd.read_json(p, orient="records", lines=False)
    else:
        df = pd.read_csv(p)  # デフォルト
    # 必須列の存在チェック
    required = ["id","title","authors","year","doi","url"]
    for c in required:
        if c not in df.columns:
            df[c] = ""
    if "tags" not in df.columns:
        df["tags"] = ""
    return df
