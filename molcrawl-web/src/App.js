import React, { useState, useEffect } from 'react';
import './App.css';

// APIから実際のディレクトリ構造を取得
const fetchDirectoryStructure = async (path = null, recursive = false, maxDepth = 3) => {
  let url;
  if (path) {
    url = `/api/directory/expand?path=${encodeURIComponent(path)}&recursive=${recursive}&maxDepth=${maxDepth}`;
  } else {
    url = '/api/directory';
  }
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || 'APIエラーが発生しました');
  }
  
  return result.data;
};

// 完全なディレクトリツリーを取得
const fetchFullDirectoryTree = async (maxDepth = 5, includeFiles = true) => {
  const url = `/api/directory/tree?maxDepth=${maxDepth}&includeFiles=${includeFiles}`;
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || 'APIエラーが発生しました');
  }
  
  return result.data;
};

// ファイルサイズのフォーマット
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

// ディレクトリツリーコンポーネント
const DirectoryTree = ({ data, expandedDirs, onToggle, level = 0 }) => {
  const indent = level * 20;

  if (!data) return null;

  const renderItem = (item, index) => {
    const isExpanded = expandedDirs.has(item.path);
    const isDirectory = item.type === 'directory';

    return (
      <div key={`${item.path}-${index}`} className="tree-item">
        <div 
          className={`tree-node ${isDirectory ? 'directory' : 'file'}`}
          style={{ paddingLeft: `${indent}px` }}
        >
          {isDirectory ? (
            <div 
              className="directory-header"
              onClick={() => onToggle(item.path, item)}
            >
              <span className="tree-icon">
                {isExpanded ? '▼' : '▶'}
              </span>
              <span className="item-icon">📁</span>
              <span className="item-name">
                {item.name}
                <span className="item-count"> ({item.count} 項目)</span>
              </span>
              {item.size > 0 && (
                <span className="item-size">{formatFileSize(item.size)}</span>
              )}
            </div>
          ) : (
            <div className="file-header">
              <span className="tree-icon-spacer"></span>
              <span className="item-icon">📄</span>
              <span className="item-name">{item.name}</span>
              <span className="item-size">{formatFileSize(item.size)}</span>
            </div>
          )}

          {isDirectory && isExpanded && item.children && (
            <DirectoryTree
              data={item.children}
              expandedDirs={expandedDirs}
              onToggle={onToggle}
              level={level + 1}
            />
          )}
        </div>
      </div>
    );
  };

  if (Array.isArray(data)) {
    return (
      <div className="directory-tree">
        {data.map(renderItem)}
      </div>
    );
  }

  return (
    <div className="directory-tree">
      {renderItem(data, 0)}
    </div>
  );
};

function App() {
  const [directoryData, setDirectoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDirs, setExpandedDirs] = useState(new Set());
  const [expandingDirs, setExpandingDirs] = useState(new Set());
  const [viewMode, setViewMode] = useState('lazy'); // 'lazy' | 'recursive' | 'full'
  const [maxDepth, setMaxDepth] = useState(3);

  // 初期データの読み込み
  const loadInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDirectoryStructure();
      setDirectoryData(data);
    } catch (err) {
      console.error('初期データ読み込みエラー:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 完全なツリーの読み込み
  const loadFullTree = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchFullDirectoryTree(maxDepth, true);
      setDirectoryData(data);
      setViewMode('full');
    } catch (err) {
      console.error('完全ツリー読み込みエラー:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 再帰的展開モードの切り替え
  const toggleRecursiveMode = async () => {
    if (viewMode === 'recursive') {
      setViewMode('lazy');
      loadInitialData();
    } else {
      setViewMode('recursive');
      setExpandedDirs(new Set());
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // ディレクトリの展開/折りたたみ
  const handleToggleDirectory = async (path, item) => {
    const newExpandedDirs = new Set(expandedDirs);
    
    if (expandedDirs.has(path)) {
      // 折りたたみ
      newExpandedDirs.delete(path);
      setExpandedDirs(newExpandedDirs);
    } else {
      // 展開
      newExpandedDirs.add(path);
      setExpandedDirs(newExpandedDirs);
      
      // 子要素がまだ読み込まれていない場合は読み込む
      if (item.children.length === 0 && item.count > 0 && viewMode !== 'full') {
        setExpandingDirs(new Set([...expandingDirs, path]));
        
        try {
          const isRecursive = viewMode === 'recursive';
          const children = await fetchDirectoryStructure(path, isRecursive, maxDepth);
          
          // directoryDataを更新
          const updateChildren = (data) => {
            if (data.path === path) {
              return { ...data, children: Array.isArray(children) ? children : [children] };
            }
            if (data.children) {
              return {
                ...data,
                children: data.children.map(updateChildren)
              };
            }
            return data;
          };
          
          setDirectoryData(prevData => updateChildren(prevData));
        } catch (err) {
          console.error('子ディレクトリ読み込みエラー:', err);
          // エラーの場合は展開状態を元に戻す
          newExpandedDirs.delete(path);
          setExpandedDirs(newExpandedDirs);
        } finally {
          setExpandingDirs(new Set([...expandingDirs].filter(p => p !== path)));
        }
      }
    }
  };

  // リフレッシュ
  const handleRefresh = () => {
    setExpandedDirs(new Set());
    if (viewMode === 'full') {
      loadFullTree();
    } else {
      loadInitialData();
    }
  };

  if (loading) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🧬 MolCrawl Dataset Browser</h1>
          <p>Fundamental Models Dataset Explorer</p>
        </header>
        <main className="App-main">
          <div className="directory-browser">
            <div className="loading">
              <span>⏳</span>
              <span>ディレクトリ構造を読み込み中...</span>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>🧬 MolCrawl Dataset Browser</h1>
          <p>Fundamental Models Dataset Explorer</p>
        </header>
        <main className="App-main">
          <div className="directory-browser">
            <div className="error">
              <span>❌ エラーが発生しました</span>
              <span>{error}</span>
              <button onClick={handleRefresh}>再試行</button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧬 MolCrawl Dataset Browser</h1>
        <p>Fundamental Models Dataset Explorer - 再帰的ディレクトリ探索</p>
      </header>
      <main className="App-main">
        <div className="directory-browser">
          <div className="tree-container">
            <div className="tree-header">
              <div className="controls">
                <button 
                  className={`mode-btn ${viewMode === 'lazy' ? 'active' : ''}`}
                  onClick={() => {setViewMode('lazy'); handleRefresh();}}
                  title="必要に応じて読み込み"
                >
                  💤 遅延
                </button>
                <button 
                  className={`mode-btn ${viewMode === 'recursive' ? 'active' : ''}`}
                  onClick={toggleRecursiveMode}
                  title="展開時に子ディレクトリも読み込み"
                >
                  🔄 再帰
                </button>
                <button 
                  className={`mode-btn ${viewMode === 'full' ? 'active' : ''}`}
                  onClick={loadFullTree}
                  title="全体を一度に読み込み"
                >
                  🌳 完全
                </button>
                <select 
                  value={maxDepth} 
                  onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                  className="depth-select"
                  title="最大読み込み深度"
                >
                  <option value={2}>深度 2</option>
                  <option value={3}>深度 3</option>
                  <option value={4}>深度 4</option>
                  <option value={5}>深度 5</option>
                  <option value={10}>深度 10</option>
                </select>
                <button className="refresh-btn" onClick={handleRefresh}>
                  🔄
                </button>
              </div>
            </div>
            <div className="mode-info">
              <span className={`mode-indicator mode-${viewMode}`}>
                {viewMode === 'lazy' && '💤 遅延読み込みモード: クリック時に個別読み込み'}
                {viewMode === 'recursive' && '🔄 再帰読み込みモード: 展開時に子ディレクトリも読み込み'}
                {viewMode === 'full' && '🌳 完全読み込みモード: 全体構造を一度に表示'}
              </span>
            </div>
            <DirectoryTree
              data={directoryData}
              expandedDirs={expandedDirs}
              onToggle={handleToggleDirectory}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
