# Streamlit App Template

このテンプレは、VS Code / GitHub / Streamlit Cloud での公開を想定した最小構成です。  
すぐに `streamlit run app.py` でローカル動作確認ができ、`requirements.txt` を元に本番環境も自動構築されます。

## 使い方（ローカル）
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 使い方（GitHub → Streamlit Cloud）
1. このテンプレートをベースに新規リポジトリを作成（またはzipを展開してpush）
2. https://streamlit.io/cloud で「New app」→ リポジトリ / ブランチ / `app.py` を指定して Deploy
3. 以後は push する度に自動デプロイ

## 構成
```
.streamlit/config.toml    # テーマなど
.vscode/launch.json       # VS Codeデバッグ（F5）
src/                      # ロジックやデータ処理関数
pages/                    # マルチページ用（任意）
app.py                    # エントリポイント
requirements.txt          # 依存（本番のビルドにも使用）
runtime.txt               # Pythonバージョン固定（任意）
.gitignore                # ゴミ除外 & secrets除外
pyproject.toml            # Ruff/mypyなどの設定（任意）
.pre-commit-config.yaml   # 保存前チェック（任意）
Makefile                  # よく使うコマンド（任意）
```
