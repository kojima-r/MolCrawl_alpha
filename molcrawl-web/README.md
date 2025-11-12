# MolCrawl Web

MolCrawl Dataset Browser - データセットを探索するためのWebインターフェース

## 必須要件

- Node.js 18以上
- npm
- Python 3.x
- **LEARNING_SOURCE_DIR環境変数の設定（必須）**

## クイックスタート

### 1. 依存関係のインストール

```bash
cd molcrawl-web
npm install
```

### 2. 使用可能なディレクトリを確認

```bash
npm run check-env
```

エラーが表示された場合、使用可能な`learning_source`ディレクトリが表示されます。

### 3. サーバーを起動

```bash
# 環境変数を設定して起動
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

または

```bash
# 環境変数をエクスポートして起動
export LEARNING_SOURCE_DIR="learning_source_202508"
npm run dev
```

### 4. ブラウザでアクセス

- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:3001/api/health

## NPMスクリプト

### 開発用

- `npm run dev` - フロントエンドとバックエンドを同時起動（推奨）
- `npm start` - フロントエンドのみ起動
- `npm run server` - バックエンドのみ起動
- `npm run check-env` - 環境変数と設定を確認

### ビルド・テスト

- `npm run build` - プロダクションビルド
- `npm test` - テスト実行
- `npm run prod` - ビルド後にサーバー起動

### コード品質

- `npm run lint` - ESLintでコードチェック
- `npm run lint:fix` - ESLintで自動修正
- `npm run format` - Prettierでフォーマット
- `npm run format:check` - フォーマットチェックのみ

## 環境変数

### LEARNING_SOURCE_DIR（必須）

データセットのルートディレクトリを指定します。

```bash
export LEARNING_SOURCE_DIR="learning_source_202508"
```

使用可能なディレクトリ:
- `learning_source_202508`
- `learning_source_20251006_genome_all`
- `learning_source_20251020-molecule-nl`

### 永続的に設定する場合

`~/.bashrc`または`~/.zshrc`に追加:

```bash
export LEARNING_SOURCE_DIR="learning_source_202508"
```

設定を反映:

```bash
source ~/.bashrc  # または source ~/.zshrc
```

## トラブルシューティング

詳細は[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)を参照してください。

### よくある問題

#### サーバーが起動しない

```bash
❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!
```

**解決方法**: 環境変数を設定してください

```bash
LEARNING_SOURCE_DIR="learning_source_202508" npm run dev
```

#### 500エラーが発生する

バックエンドサーバーが起動していない可能性があります。

**解決方法**: `npm start`ではなく`npm run dev`を使用してください

## プロジェクト構造

```
molcrawl-web/
├── api/                    # バックエンドAPI
│   ├── directory.js       # ディレクトリAPI
│   ├── genome-species.js  # ゲノム種API
│   └── zinc-checker.js    # ZINC20データチェッカー
├── src/                    # Reactフロントエンド
│   ├── App.js             # メインアプリケーション
│   ├── ExperimentDashboard.js
│   ├── GenomeSpeciesList.js
│   └── ZincChecker.js
├── public/                 # 静的ファイル
├── server.js              # Expressサーバー
├── package.json           # 依存関係
└── check-config.js        # 設定チェックスクリプト
```

## API エンドポイント

### ヘルスチェック
- `GET /api/health` - サーバーステータス

### ディレクトリ操作
- `GET /api/directory` - ルートディレクトリ構造取得
- `GET /api/directory/expand?path=<path>` - ディレクトリ展開
- `GET /api/directory/tree?maxDepth=5` - 完全ツリー取得

### ゲノムデータ
- `GET /api/genome/species` - ゲノム種リスト取得
- `GET /api/genome/species/category?category=<category>` - カテゴリ別種リスト

### ZINC20データ
- `GET /api/zinc/check` - ZINC20データチェック
- `GET /api/zinc/count` - ZINC20データ件数取得

## 開発

### ESLint設定

`.eslintrc.json`に設定があります。詳細は[ESLINT_SETUP.md](./ESLINT_SETUP.md)を参照。

### Prettier設定

`.prettierrc.json`に設定があります。
