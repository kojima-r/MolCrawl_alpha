import React, { useState, useEffect } from 'react';
import './GPUResources.css';

const GPUResources = () => {
    const [gpuInfo, setGpuInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(false);
    const [refreshInterval, setRefreshInterval] = useState(5000); // 5秒

    // GPU情報を取得
    const fetchGpuInfo = async () => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch('/api/gpu/info');
            const data = await response.json();

            if (data.success) {
                setGpuInfo(data.data);
            } else {
                setError(data.error || 'GPU情報の取得に失敗しました');
                setGpuInfo(null);
            }
        } catch (err) {
            console.error('GPU info fetch error:', err);
            setError('GPU情報の取得中にエラーが発生しました: ' + err.message);
            setGpuInfo(null);
        } finally {
            setLoading(false);
        }
    };

    // 初回読み込み
    useEffect(() => {
        fetchGpuInfo();
    }, []);

    // 自動更新
    useEffect(() => {
        if (autoRefresh) {
            const intervalId = setInterval(() => {
                fetchGpuInfo();
            }, refreshInterval);

            return () => clearInterval(intervalId);
        }
    }, [autoRefresh, refreshInterval]);

    const handleRefresh = () => {
        fetchGpuInfo();
    };

    const toggleAutoRefresh = () => {
        setAutoRefresh(!autoRefresh);
    };

    const handleIntervalChange = (e) => {
        setRefreshInterval(parseInt(e.target.value));
    };

    if (loading && !gpuInfo) {
        return (
            <div className="gpu-resources">
                <div className="gpu-header">
                    <h2>🖥️ GPUリソース情報</h2>
                </div>
                <div className="gpu-loading">
                    <span>⏳</span>
                    <span>GPU情報を読み込み中...</span>
                </div>
            </div>
        );
    }

    if (error && !gpuInfo) {
        return (
            <div className="gpu-resources">
                <div className="gpu-header">
                    <h2>🖥️ GPUリソース情報</h2>
                    <div className="gpu-controls">
                        <button onClick={handleRefresh} className="refresh-button">
                            🔄 再試行
                        </button>
                    </div>
                </div>
                <div className="gpu-error">
                    <span>❌ {error}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="gpu-resources">
            <div className="gpu-header">
                <h2>🖥️ GPUリソース情報</h2>
                <div className="gpu-controls">
                    <label className="auto-refresh-control">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={toggleAutoRefresh}
                        />
                        自動更新
                    </label>
                    {autoRefresh && (
                        <select
                            value={refreshInterval}
                            onChange={handleIntervalChange}
                            className="interval-select"
                        >
                            <option value={2000}>2秒</option>
                            <option value={5000}>5秒</option>
                            <option value={10000}>10秒</option>
                            <option value={30000}>30秒</option>
                        </select>
                    )}
                    <button
                        onClick={handleRefresh}
                        className="refresh-button"
                        disabled={loading}
                    >
                        {loading ? '⏳' : '🔄'} 更新
                    </button>
                </div>
            </div>

            {gpuInfo && (
                <div className="gpu-content">
                    <div className="gpu-timestamp">
                        最終更新: {new Date(gpuInfo.timestamp).toLocaleString('ja-JP')}
                    </div>
                    <div className="nvidia-smi-output">
                        <pre>{gpuInfo.raw}</pre>
                    </div>
                </div>
            )}

            {loading && gpuInfo && (
                <div className="gpu-updating">更新中...</div>
            )}
        </div>
    );
};

export default GPUResources;
