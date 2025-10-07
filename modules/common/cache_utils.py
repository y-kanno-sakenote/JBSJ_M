# modules/common/cache_utils.py
# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def _sig_from_params(*parts) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

def cache_csv_path(prefix: str, *params) -> Path:
    sig = _sig_from_params(*params)
    return CACHE_DIR / f"{prefix}_{sig}.csv"

def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def save_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass