const { exec } = require('child_process');
const util = require('util');

const execPromise = util.promisify(exec);

/**
 * nvidia-smiコマンドを実行してGPU情報を取得
 */
const getNvidiaSmiInfo = async () => {
    try {
        // nvidia-smiコマンドを実行
        const { stdout, stderr } = await execPromise('nvidia-smi');

        if (stderr && !stdout) {
            throw new Error(stderr);
        }

        return {
            success: true,
            data: {
                raw: stdout,
                timestamp: new Date().toISOString()
            }
        };
    } catch (error) {
        console.error('nvidia-smi execution error:', error);

        // nvidia-smiが利用できない場合のエラーハンドリング
        if (error.code === 'ENOENT' || error.message.includes('not found')) {
            return {
                success: false,
                error: 'nvidia-smi コマンドが見つかりません。NVIDIA GPUまたはNVIDIAドライバーがインストールされていない可能性があります。',
                timestamp: new Date().toISOString()
            };
        }

        return {
            success: false,
            error: error.message || 'nvidia-smi実行中にエラーが発生しました',
            timestamp: new Date().toISOString()
        };
    }
};

/**
 * nvidia-smiのXML形式で詳細情報を取得
 */
const getNvidiaSmiXml = async () => {
    try {
        const { stdout, stderr } = await execPromise('nvidia-smi -q -x');

        if (stderr && !stdout) {
            throw new Error(stderr);
        }

        return {
            success: true,
            data: {
                xml: stdout,
                timestamp: new Date().toISOString()
            }
        };
    } catch (error) {
        console.error('nvidia-smi XML execution error:', error);
        return {
            success: false,
            error: error.message || 'nvidia-smi XML取得中にエラーが発生しました',
            timestamp: new Date().toISOString()
        };
    }
};

/**
 * APIエンドポイント: nvidia-smi出力を取得
 */
const getGpuInfo = async (req, res) => {
    try {
        const result = await getNvidiaSmiInfo();

        if (result.success) {
            res.json(result);
        } else {
            res.status(500).json(result);
        }
    } catch (error) {
        console.error('GPU info API error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'GPU情報の取得に失敗しました',
            timestamp: new Date().toISOString()
        });
    }
};

/**
 * APIエンドポイント: nvidia-smi XML出力を取得
 */
const getGpuXmlInfo = async (req, res) => {
    try {
        const result = await getNvidiaSmiXml();

        if (result.success) {
            res.json(result);
        } else {
            res.status(500).json(result);
        }
    } catch (error) {
        console.error('GPU XML info API error:', error);
        res.status(500).json({
            success: false,
            error: error.message || 'GPU XML情報の取得に失敗しました',
            timestamp: new Date().toISOString()
        });
    }
};

module.exports = {
    getGpuInfo,
    getGpuXmlInfo
};
