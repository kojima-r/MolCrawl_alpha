# MolCrawl Web - トラブルシューティングガイド

## 必須: 環境変数の設定

**重要**: このアプリケーションは`LEARNING_SOURCE_DIR`環境変数が必須です。設定しないと起動しません。

## 起動方法

### ステップ1: 使用可能なディレクトリを確認

```bash
cd molcrawl-web
npm run check-env
```

これにより、使用可能な`learning_source`ディレクトリが表示されます。

### ステップ2: 環境変数を設定して起動

#### 方法1: 1行で起動（推奨）

```bash
cd molcrawl-web
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

#### 方法2: 環境変数をエクスポートして起動

```bash
cd molcrawl-web
export LEARNING_SOURCE_DIR="learning_source_202508"
npm run dev
```

#### 方法3: .bashrcまたは.zshrcに追加（永続化）

```bash
# ~/.bashrc または ~/.zshrc に追加
export LEARNING_SOURCE_DIR="learning_source_202508"

# 設定を反映
source ~/.bashrc  # または source ~/.zshrc
```

## よくあるエラーと解決方法

### エラー: `LEARNING_SOURCE_DIR environment variable is required!`

環境変数が設定されていません。

**解決方法**:
```bash
# 使用可能なディレクトリを確認
npm run check-env

# 環境変数を設定して起動
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

### エラー: `Specified LEARNING_SOURCE_DIR does not exist!`

指定したディレクトリが存在しません。

**解決方法**:
```bash
# 正しいディレクトリ名を確認
ls -d ../learning_source*

# 正しい名前で再起動
LEARNING_SOURCE_DIR="正しいディレクトリ名" npm run dev
```

### エラー: `ECONNREFUSED localhost:3001`

バックエンドサーバーが起動していません。

**解決方法**:
- `npm start`ではなく`npm run dev`を使用してください
- または別ターミナルで`npm run server`を実行してください

### ポートが既に使用されている

```bash
# ポート3001を使用しているプロセスを確認
lsof -i :3001

# プロセスを終了
kill -9 <PID>
```

## アクセスURL

起動後、以下のURLにアクセスできます：

- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:3001/api/health
- **ディレクトリAPI**: http://localhost:3001/api/directory
