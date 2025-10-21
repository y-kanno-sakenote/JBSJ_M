# modules/analysis/targettype_entry.py
# -*- coding: utf-8 -*-
"""
ターゲットタイプ（対象物・研究タイプ）タブのエントリーポイント。
モジュール分割版（modules/analysis/targettype/）から安全にインポートして公開します。
"""
from __future__ import annotations
import streamlit as st

try:
    from modules.analysis.targettype import render_targettype_tab  # type: ignore
except Exception:
    def render_targettype_tab(_df):
        st.warning("targettype タブの読み込みに失敗しました。コードを確認してください。")