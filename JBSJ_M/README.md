# JBSJ_M リポジトリ - 整理記録

このリポジトリでは、開発中に生成されたキャッシュやバイトコードなどの生成物を Git 管理から除外し、作業ツリーをクリーンにする整理を行いました。

日付: 2025-10-21

## 実施した操作（要約）
- `.gitignore` を追加し、以下を無視対象にしました: `__pycache__/`, `*.pyc`, `cache/`, `.cache/`, `archive/`, `.DS_Store` など。
- 既に Git 管理下にあった生成CSV（`cache/` と `.cache/` 内のファイル）を `archive/cache_backup/` に移動してコミットしました。これにより大きな生成物はリポジトリに残りませんが、ローカルには保存されています。
- `modules/` 以下の Python バイトコード（`__pycache__` ディレクトリ）をディスクから削除しました（再生成可能）。

## 変更された主なコミット
- chore: add .gitignore and untrack cache and pyc files
- chore: archive cache files to archive/cache_backup
- chore: remove __pycache__ directories from disk

## 保管場所（ローカル）
- 生成CSV 等は `archive/cache_backup/` に移動されています。必要であればここから復元できます。

## 復元方法

1. もし `archive/cache_backup/` から `cache/` に戻したい場合（ローカル復元）:

```bash
# 移動（上書きに注意）
mkdir -p cache
mv archive/cache_backup/* cache/
```

2. もしコミット前の状態（リポジトリ内にあったファイル）を履歴から取り出したい場合:

```bash
# 削除前コミットのハッシュを確認
git log --pretty=oneline
# 例: git checkout <COMMIT_SHA> -- path/to/file.csv
git checkout <COMMIT_SHA> -- cache/coauthor_edges_*.csv
```

3. `__pycache__` は Python 実行時に自動で再生成されます。不要であればそのまま無視してください。

## 開発／起動（簡易）
リポジトリの Python 要件は `requirements.txt` にあります。Streamlit アプリを起動する場合の例:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

（環境や Python バージョンに依存します。必要に応じて `pip` のインストール先やバージョンを調整してください。）

## 注意点
- `archive/` は `.gitignore` に含めているため、他の開発者がリポジトリをクローンしても `archive/` の中身は取得できません。重要な生成物を共有する場合は別途共有方法（外部ストレージや専用データリポジトリ）を検討してください。
- Google Drive 等の同期フォルダで作業している場合、ファイル操作がクラウドに反映される点に注意してください。

## 次のステップ候補
- ドキュメントの追加（本 README を拡充）
- `modules/` を `src/` に移動するリファクタ（案B）を別ブランチで段階的に進める
- テストスイートの追加

---
もし README の内容に追記したい点があれば指示してください。必要ならこの README を拡張して、復元スクリプトや細かい注意事項を追加します。
