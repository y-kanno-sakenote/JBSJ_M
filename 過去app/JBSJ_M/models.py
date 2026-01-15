# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paper:
    id: str
    year: int
    volume: Optional[str]
    issue: Optional[str]
    start_page: Optional[str]
    end_page: Optional[str]
    title: str
    authors: str  # セミコロン区切りを推奨
    publish_date: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    tags: str = ""
