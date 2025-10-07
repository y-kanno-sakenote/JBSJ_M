# modules/common/cache_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import Any, Iterable, Tuple, Optional

import pandas as pd

# 新: 非隠しディレクトリに統一
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# 互換: 旧 .cache も参照（読み込み時のみ）
DOT_CACHE_DIR = Path(".cache")
# 旧ディレクトリがあるなら作成だけはしておく（権限エラー回避用）
DOT_CACHE_DIR.mkdir(exist_ok=True)

def _sig_from_params(*parts: Any) -> str:
    """引数から安定ハッシュ生成。順序と値が同じなら同じキーになる。"""
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

def _csv_name(prefix: str, sig: str) -> str:
    return f"{prefix}_{sig}.csv"

def _pkl_name(prefix: str, sig: str) -> str:
    return f"{prefix}_{sig}.pkl"

# ========== CSV ==========

def cache_csv_path(prefix: str, *params: Any) -> Path:
    """cache/ 側のCSVパスを返す（書き込み先）。"""
    sig = _sig_from_params(*params)
    return CACHE_DIR / _csv_name(prefix, sig)

def _legacy_csv_path(prefix: str, *params: Any) -> Path:
    """互換: .cache 側のCSVパス（読み込みフォールバック用）。"""
    sig = _sig_from_params(*params)
    return DOT_CACHE_DIR / _csv_name(prefix, sig)

def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """
    cache/ を優先して読み込む。
    無ければ .cache/ を探して読み込み、見つかれば cache/ にコピーして移行。
    """
    # まず cache/ を見る
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    # 互換: .cache/ を見る
    legacy_path = DOT_CACHE_DIR / path.name
    if legacy_path.exists():
        try:
            df = pd.read_csv(legacy_path)
            # 見つかったら cache/ にコピーして以降は新パスを使う
            try:
                shutil.copy2(legacy_path, path)
            except Exception:
                pass  # コピー失敗は無視（読み込みは成功している）
            return df
        except Exception:
            return None

    return None

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """cache/ にCSV保存（親ディレクトリが無ければ作成）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        pass

# ========== Pickle（任意で使いたい場合） ==========

try:
    import pickle
    HAS_PICKLE = True
except Exception:
    HAS_PICKLE = False

def cache_pkl_path(prefix: str, *params: Any) -> Path:
    sig = _sig_from_params(*params)
    return CACHE_DIR / _pkl_name(prefix, sig)

def _legacy_pkl_path(prefix: str, *params: Any) -> Path:
    sig = _sig_from_params(*params)
    return DOT_CACHE_DIR / _pkl_name(prefix, sig)

def load_pkl_if_exists(path: Path) -> Any | None:
    if not HAS_PICKLE:
        return None
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    legacy_path = DOT_CACHE_DIR / path.name
    if legacy_path.exists():
        try:
            with open(legacy_path, "rb") as f:
                obj = pickle.load(f)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_path, path)
            except Exception:
                pass
            return obj
        except Exception:
            return None
    return None

def save_pkl(obj: Any, path: Path) -> None:
    if not HAS_PICKLE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

# ========== ユーティリティ（任意） ==========

def clear_cache(pattern: str | None = None) -> int:
    """
    cache/ のファイルを削除（pattern があればワイルドカードで絞り込み）。
    返り値: 削除数
    """
    cnt = 0
    if not CACHE_DIR.exists():
        return 0
    if pattern:
        paths = list(CACHE_DIR.glob(pattern))
    else:
        paths = list(CACHE_DIR.glob("*"))
    for p in paths:
        try:
            p.unlink()
            cnt += 1
        except Exception:
            pass
    return cnt