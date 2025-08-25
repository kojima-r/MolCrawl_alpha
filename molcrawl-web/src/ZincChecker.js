import React, { useState } from 'react';
import './ZincChecker.css';

const ZincChecker = () => {
  const [checkResult, setCheckResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('ja-JP');
  };

  const checkZincFiles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/zinc/check');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'チェックに失敗しました');
      }
      
      setCheckResult(result.data);
    } catch (err) {
      console.error('ZINC チェックエラー:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (rate) => {
    if (rate >= 95) return '#28a745'; // green
    if (rate >= 80) return '#ffc107'; // yellow
    return '#dc3545'; // red
  };

  return (
    <div className="zinc-checker">
      <div className="zinc-checker-header">
        <h3>🧪 ZINC20 データ完整性チェック</h3>
        <p>download_zinc.shで定義されたファイルのダウンロード状況を確認します</p>
        <button 
          className="check-button"
          onClick={checkZincFiles}
          disabled={loading}
        >
          {loading ? '⏳ チェック中...' : '🔍 ZINC データをチェック'}
        </button>
      </div>

      {error && (
        <div className="zinc-error">
          <span>❌ エラー: {error}</span>
        </div>
      )}

      {checkResult && (
        <div className="zinc-results">
          {!checkResult.exists ? (
            <div className="zinc-not-found">
              <h4>📁 ZINC20 データディレクトリが見つかりません</h4>
              <p>パス: <code>{checkResult.path}</code></p>
              <p>download_zinc.shを実行してデータをダウンロードしてください。</p>
            </div>
          ) : (
            <>
              <div className="zinc-summary">
                <h4>📊 チェック結果サマリー</h4>
                <div className="summary-stats">
                  <div className="stat-item">
                    <span className="stat-label">総ファイル数:</span>
                    <span className="stat-value">{checkResult.total}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">存在するファイル:</span>
                    <span className="stat-value" style={{ color: getStatusColor(parseFloat(checkResult.completionRate)) }}>
                      {checkResult.existing} ({checkResult.completionRate}%)
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">データ有りファイル:</span>
                    <span className="stat-value" style={{ color: getStatusColor(parseFloat(checkResult.sizeCompletionRate)) }}>
                      {checkResult.withSize} ({checkResult.sizeCompletionRate}%)
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">総データサイズ:</span>
                    <span className="stat-value">{formatFileSize(checkResult.totalSize)}</span>
                  </div>
                </div>
              </div>

              {checkResult.directoryStats && (
                <div className="zinc-directory-stats">
                  <h4>📂 ディレクトリ別統計</h4>
                  <div className="directory-grid">
                    {Object.entries(checkResult.directoryStats).map(([dir, stats]) => (
                      <div key={dir} className="directory-stat">
                        <span className="dir-name">{dir}/</span>
                        <span className="dir-progress">
                          {stats.existing}/{stats.total}
                          <div className="progress-bar">
                            <div 
                              className="progress-fill" 
                              style={{ 
                                width: `${(stats.existing / stats.total) * 100}%`,
                                backgroundColor: getStatusColor((stats.existing / stats.total) * 100)
                              }}
                            ></div>
                          </div>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {checkResult.missing && checkResult.missing.length > 0 && (
                <div className="zinc-missing">
                  <h4>⚠️ 不足ファイル ({checkResult.missing.length}個)</h4>
                  <div className="missing-files">
                    {checkResult.missing.slice(0, 20).map((file, index) => (
                      <code key={index} className="missing-file">{file}</code>
                    ))}
                    {checkResult.missing.length > 20 && (
                      <p>...他 {checkResult.missing.length - 20} ファイル</p>
                    )}
                  </div>
                </div>
              )}

              <div className="zinc-meta">
                <p>💾 パス: <code>{checkResult.path}</code></p>
                <p>🕒 最終チェック: {formatDate(checkResult.lastChecked)}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ZincChecker;
