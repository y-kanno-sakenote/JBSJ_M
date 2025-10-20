from __future__ import annotations
import streamlit as st

def expander(title: str, items: list[str], height: int = 200) -> None:
    if not items: return
    with st.expander(title, expanded=False):
        _grid(items, height=height)

def _grid(items: list[str], height: int = 200) -> None:
    if not items: return
    html = """
    <style>
      .copy-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px,1fr)); gap:6px; }
      .copy-chip { display:flex; align-items:center; justify-content:space-between;
                   padding:4px 8px; background:#f5f5f7; border:1px solid #ddd; border-radius:8px; font-size:12px; }
      .copy-chip button { border:none; background:#e9e9ee; padding:3px 6px; border-radius:6px; cursor:pointer; }
      .copy-chip button:hover { background:#dcdce3; }
    </style>
    <div class="copy-grid">
    """
    for name in items:
        safe_text = str(name).replace("\\", "\\\\").replace("'", "\\'")
        html += f"""
        <div class="copy-chip">
          <span>{safe_text}</span>
          <button onclick="navigator.clipboard.writeText('{safe_text}');
                           const n=document.createElement('div');
                           n.textContent='📋「{safe_text}」をコピーしました';
                           n.style='position:fixed;bottom:80px;right:30px;padding:10px 18px;background:#333;color:#fff;border-radius:8px;opacity:.94;font-size:13px;z-index:9999';
                           document.body.appendChild(n); setTimeout(()=>n.remove(),1400);">📋</button>
        </div>
        """
    html += "</div>"
    import streamlit.components.v1 as components
    components.html(html, height=height, scrolling=True)