const express = require('express');
const cors = require('cors');
const path = require('path');

// 環境変数チェック（API読み込み前に実行）
if (!process.env.LEARNING_SOURCE_DIR) {
  console.error('');
  console.error('❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!');
  console.error('');
  console.error('Please set it before starting the server:');
  console.error('  export LEARNING_SOURCE_DIR="learning_source_202508"');
  console.error('  npm run dev');
  console.error('');
  console.error('Or run with inline environment variable:');
  console.error('  LEARNING_SOURCE_DIR="learning_source_202508" npm run dev');
  console.error('');
  process.exit(1);
}

const { getDirectoryStructure, expandDirectory, getFullDirectoryTree, checkZincData, getZincDataCounts } = require('./api/directory');
const { getGenomeSpeciesList, getGenomeSpeciesByCategory } = require('./api/genome-species');
const datasetProgressRouter = require('./api/dataset-progress');
const gpt2TrainingStatusRouter = require('./api/gpt2-training-status');

const app = express();
const PORT = process.env.PORT || 3001;

// model_dirの値をサーバー起動時に確認
console.log('✅ Server starting with configuration:');
console.log('   LEARNING_SOURCE_DIR:', process.env.LEARNING_SOURCE_DIR);
console.log('   Working directory:', process.cwd());

// CORS設定
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  methods: ['GET', 'POST'],
  credentials: true
}));

// JSON解析
app.use(express.json());

// 静的ファイル配信（現在のディレクトリから）
app.use(express.static(__dirname));

// API Routes
app.get('/api/directory', getDirectoryStructure);
app.get('/api/directory/expand', expandDirectory);
app.get('/api/directory/tree', getFullDirectoryTree);
app.get('/api/zinc/check', checkZincData);
app.get('/api/zinc/count', getZincDataCounts);
app.get('/api/genome/species', getGenomeSpeciesList);
app.get('/api/genome/species/category', getGenomeSpeciesByCategory);
app.use('/api/dataset-progress', datasetProgressRouter);
app.use('/api/gpt2-training-status', gpt2TrainingStatusRouter);

// ヘルスチェック
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    message: 'MolCrawl Web API Server is running',
    endpoints: [
      '/api/directory - ルートディレクトリ構造取得',
      '/api/directory/expand?path=<path>&recursive=true - ディレクトリ展開',
      '/api/directory/tree?maxDepth=5&includeFiles=true - 完全ツリー取得',
      '/api/zinc/check - ZINC20データチェック',
      '/api/zinc/count - ZINC20データ件数取得',
      '/api/genome/species - ゲノム種リスト取得',
      '/api/genome/species/category?category=<category> - カテゴリ別種リスト取得',
      '/api/dataset-progress - 全データセット準備進捗取得',
      '/api/dataset-progress/:datasetKey - 特定データセット詳細進捗取得',
      '/api/gpt2-training-status - 全GPT-2モデルの学習状況取得',
      '/api/gpt2-training-status/:dataset - 特定データセットのGPT-2学習状況',
      '/api/gpt2-training-status/:dataset/:size - 特定モデルの詳細情報'
    ]
  });
});

// エラーハンドリング
app.use((error, req, res, next) => {
  console.error('Server Error:', error);
  res.status(500).json({
    error: 'Internal Server Error',
    message: error.message,
    timestamp: new Date().toISOString()
  });
});

// 404ハンドリング
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.url} not found`,
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📁 Directory API: http://localhost:${PORT}/api/directory`);
  console.log(`🌳 Full Tree API: http://localhost:${PORT}/api/directory/tree`);
  console.log(`🌐 Test page: http://localhost:${PORT}/test.html`);
});

module.exports = app;
