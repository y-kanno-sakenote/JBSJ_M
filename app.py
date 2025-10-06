import streamlit as st
import pandas as pd

st.set_page_config(page_title="My Streamlit App", layout="wide")
st.title("Hello, Streamlit 👋")
st.write("最初の雛形です。`src/` に処理関数、`pages/` に追加ページを置けます。")

@st.cache_data
def get_sample_df() -> pd.DataFrame:
    return pd.DataFrame({"銘柄": ["鳴門鯛", "勝瑞城", "松竹梅"], "酸度": [1.4, 1.6, 1.2]})

df = get_sample_df()
st.subheader("サンプルデータ")
st.dataframe(df, use_container_width=True)

with st.expander("使い方"):
    st.markdown("""
    1. `src/` に関数を追加
    2. `pages/` に `1_...py` のようなファイルを置くとタブに表示
    3. GitHub へ push → Streamlit Cloud で `New app` から公開
    """)
