from __future__ import annotations
from pathlib import Path
import streamlit as st

def get_japanese_font_path() -> str | None:
    for p in ["fonts/IPAexGothic.ttf",
              "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
              "/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]:
        if Path(p).exists(): return p
    return None

def safe_show_image(obj):
    import io, numpy as np
    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore

    if obj is None:
        st.warning("画像データが None でした。"); return

    try:
        import matplotlib.figure
        if isinstance(obj, matplotlib.figure.Figure):
            st.pyplot(obj); return
    except Exception:
        pass

    def _st_image_compat(data):
        """Call st.image with use_container_width when supported; fall back to simple call otherwise."""
        try:
            # Preferred modern API
            st.image(data, use_container_width=True)
        except TypeError:
            # Older Streamlit versions may not accept use_container_width
            try:
                st.image(data)
            except Exception as e:
                raise

    if Image is not None and isinstance(obj, Image.Image):
        try:
            img = obj.convert("RGBA") if obj.mode not in ("RGB","RGBA") else obj
            buf = io.BytesIO(); img.save(buf, format="PNG")
            try:
                _st_image_compat(buf.getvalue())
            except Exception as e:
                st.warning(f"PIL画像の表示で例外: {e!s}")
        except Exception as e:
            st.warning(f"PIL画像の処理で例外: {e!s}")
        return

    if isinstance(obj, np.ndarray):
        a = obj
        if a.dtype in (np.float32, np.float64):
            if np.nanmax(a) <= 1.0: a = (np.nan_to_num(a)*255.0).clip(0,255).astype(np.uint8)
            else: a = np.nan_to_num(a).clip(0,255).astype(np.uint8)
        elif a.dtype != np.uint8:
            a = np.nan_to_num(a).clip(0,255).astype(np.uint8)
        try:
            _st_image_compat(a)
        except Exception as e:
            st.warning(f"配列画像の表示で例外: {e!s}")
        return

    if isinstance(obj, (bytes, bytearray)):
        try:
            _st_image_compat(obj)
        except Exception as e:
            st.warning(f"バイト画像の表示で例外: {e!s}")
        return
    if isinstance(obj, str):
        try:
            _st_image_compat(obj)
        except Exception as e:
            st.warning(f"画像URL/パスの表示で例外: {e!s}")
        return

    st.warning(f"st.imageが扱えない型: {type(obj)}")