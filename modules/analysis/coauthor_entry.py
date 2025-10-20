# modules/analysis/coauthor_entry.py
# -*- coding: utf-8 -*-
"""
研究者（共著）タブのエントリーポイント。
モジュール分割版（modules/analysis/coauthor/）から安全にインポートして公開します。
"""
from __future__ import annotations
import streamlit as st

try:
    # 分割後の本体（すでに作った coauthor 側）
    from modules.analysis.coauthor import render_coauthor_tab  # type: ignore
except Exception:
    # 失敗してもアプリは落とさず注意だけ出す
    def render_coauthor_tab(_df):
        st.warning("coauthor タブの読み込みに失敗しました。コードを確認してください。")