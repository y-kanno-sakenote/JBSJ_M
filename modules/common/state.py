# modules/common/state.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class GlobalFilters:
    """共通フィルター情報を保持するデータ構造"""
    year_from: Optional[int]
    year_to: Optional[int]
    targets: List[str]
    types: List[str]