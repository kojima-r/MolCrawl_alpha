/**
 * Dataset Preparation Progress API
 * 各データセット準備スクリプトの進捗状況を監視するAPI
 */

const express = require('express');
const fs = require('fs');
const path = require('path');

const router = express.Router();

/**
 * 各データセットの定義
 * マーカーファイルと出力ファイル/ディレクトリで進捗を判定
 */
const DATASETS = {
  protein_sequence: {
    name: 'Protein Sequence (Uniprot)',
    baseDir: 'protein_sequence',
    steps: [
      {
        id: 'download',
        name: 'Uniprot Download',
        marker: 'download_complete.marker',
        checkFiles: [],
      },
      {
        id: 'fasta_to_raw',
        name: 'FASTA to Raw Conversion',
        marker: 'fasta_to_raw_complete.marker',
        checkFiles: ['raw_files'],
      },
      {
        id: 'tokenize',
        name: 'Tokenization to Parquet',
        marker: 'tokenize_to_parquet_complete.marker',
        checkFiles: ['parquet_files/train.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/protein_sequence_tokenized_lengths_dist.png',
      statistics: null,
    },
  },
  genome_sequence: {
    name: 'Genome Sequence (RefSeq)',
    baseDir: 'genome_sequence',
    steps: [
      {
        id: 'download',
        name: 'RefSeq Download',
        marker: 'download_complete.marker',
        checkFiles: [],
      },
      {
        id: 'fasta_to_raw',
        name: 'FASTA to Raw Conversion',
        marker: 'fasta_to_raw_complete.marker',
        checkFiles: ['raw_files'],
      },
      {
        id: 'train_tokenizer',
        name: 'Tokenizer Training',
        marker: 'train_tokenizer_complete.marker',
        checkFiles: ['spm_tokenizer.model'],
      },
      {
        id: 'raw_to_parquet',
        name: 'Raw to Parquet Conversion',
        marker: 'raw_to_parquet_complete.marker',
        checkFiles: ['parquet_files'],
      },
    ],
    outputs: {
      plot: '../assets/img/genome_sequence_tokenized_lengths_dist.png',
      statistics: null,
    },
  },
  rna: {
    name: 'RNA (CellxGene)',
    baseDir: 'rna',
    steps: [
      {
        id: 'build_list',
        name: 'Build Dataset List',
        marker: 'build_list_complete.marker',
        checkFiles: [],
      },
      {
        id: 'download',
        name: 'Dataset Download',
        marker: 'download_complete.marker',
        checkFiles: [],
      },
      {
        id: 'h5ad_to_loom',
        name: 'H5AD to Loom Conversion',
        marker: 'h5ad_to_loom_complete.marker',
        checkFiles: [],
      },
      {
        id: 'tokenize',
        name: 'Tokenization',
        marker: 'tokenize_complete.marker',
        checkFiles: ['parquet_files'],
      },
      {
        id: 'vocab',
        name: 'Vocabulary Generation',
        marker: null,
        checkFiles: ['gene_vocab.json'],
      },
    ],
    outputs: {
      plot: '../assets/img/rna_tokenized_lengths_dist.png',
      statistics: 'rna_stats.json',
      geneList: 'gene_list_with_stats.tsv',
    },
  },
  molecule_nl: {
    name: 'Molecule Natural Language (SMolInstruct)',
    baseDir: 'molecule_nl',
    steps: [
      {
        id: 'download',
        name: 'Dataset Download/Copy',
        marker: null,
        checkFiles: ['osunlp/SMolInstruct/dataset_info.json'],
      },
      {
        id: 'tokenize',
        name: 'Tokenization & Processing',
        marker: null,
        checkFiles: ['molecule_related_natural_language_tokenized.parquet'],
      },
    ],
    outputs: {
      plot: '../assets/img/molecule_nl_tokenized_train_lengths_dist.png',
      statistics: null,
    },
  },
  compounds: {
    name: 'Compounds (OrganiX13)',
    baseDir: 'compounds',
    steps: [
      {
        id: 'download',
        name: 'OrganiX13 Download',
        marker: 'organix13/download_complete.marker',
        checkFiles: [],
      },
      {
        id: 'tokenize',
        name: 'SMILES & Scaffolds Tokenization',
        marker: 'organix13/tokenized_complete.marker',
        checkFiles: ['organix13/OrganiX13_tokenized.parquet'],
      },
      {
        id: 'statistics',
        name: 'Statistics Generation',
        marker: 'organix13/stats_complete.marker',
        checkFiles: [],
      },
    ],
    outputs: {
      plot: '../assets/img/compounds_tokenized_SMILES_lengths_dist.png',
      scaffoldPlot: '../assets/img/compounds_tokenized_Scaffolds_lengths_dist.png',
      statistics: null,
    },
  },
};

/**
 * ファイル/ディレクトリの存在確認
 */
function checkExists(fullPath) {
  try {
    return fs.existsSync(fullPath);
  } catch (error) {
    return false;
  }
}

/**
 * ディレクトリ内にファイルが存在するか確認
 */
function checkDirHasFiles(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) return false;
    const stats = fs.statSync(dirPath);
    if (!stats.isDirectory()) return false;
    const files = fs.readdirSync(dirPath);
    return files.length > 0;
  } catch (error) {
    return false;
  }
}

/**
 * 単一ステップの状態をチェック
 */
function checkStepStatus(learningSourcePath, dataset, step) {
  const baseDir = path.join(learningSourcePath, dataset.baseDir);

  // マーカーファイルがあればそれを優先
  if (step.marker) {
    const markerPath = path.join(baseDir, step.marker);
    if (checkExists(markerPath)) {
      return 'completed';
    }
  }

  // checkFilesで詳細確認
  if (step.checkFiles && step.checkFiles.length > 0) {
    let allExist = true;
    for (const file of step.checkFiles) {
      const filePath = path.join(baseDir, file);
      const exists = checkExists(filePath);

      if (!exists) {
        allExist = false;
        break;
      }

      // ディレクトリの場合、中身があるか確認
      try {
        const stats = fs.statSync(filePath);
        if (stats.isDirectory() && !checkDirHasFiles(filePath)) {
          allExist = false;
          break;
        }
      } catch (error) {
        allExist = false;
        break;
      }
    }

    if (allExist) {
      return 'completed';
    }
  }

  return 'pending';
}

/**
 * データセットの全体進捗を取得
 */
function getDatasetProgress(learningSourcePath, datasetKey, dataset) {
  const steps = dataset.steps.map((step) => {
    const status = checkStepStatus(learningSourcePath, dataset, step);
    return {
      id: step.id,
      name: step.name,
      status: status,
    };
  });

  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const totalSteps = steps.length;
  const progressPercent = Math.round((completedSteps / totalSteps) * 100);

  // 出力ファイルの存在確認
  const outputs = {};
  const projectRoot = path.dirname(path.dirname(learningSourcePath));

  if (dataset.outputs.plot) {
    const plotPath = path.join(projectRoot, dataset.outputs.plot);
    outputs.plot = checkExists(plotPath);
  }

  if (dataset.outputs.scaffoldPlot) {
    const scaffoldPlotPath = path.join(projectRoot, dataset.outputs.scaffoldPlot);
    outputs.scaffoldPlot = checkExists(scaffoldPlotPath);
  }

  if (dataset.outputs.statistics) {
    const statsPath = path.join(
      learningSourcePath,
      dataset.baseDir,
      dataset.outputs.statistics
    );
    outputs.statistics = checkExists(statsPath);
  }

  if (dataset.outputs.geneList) {
    const geneListPath = path.join(
      learningSourcePath,
      dataset.baseDir,
      dataset.outputs.geneList
    );
    outputs.geneList = checkExists(geneListPath);
  }

  return {
    name: dataset.name,
    baseDir: dataset.baseDir,
    steps: steps,
    progress: {
      completed: completedSteps,
      total: totalSteps,
      percent: progressPercent,
    },
    outputs: outputs,
    status:
      completedSteps === totalSteps
        ? 'completed'
        : completedSteps > 0
          ? 'in_progress'
          : 'not_started',
  };
}

/**
 * GET /api/dataset-progress
 * 全データセットの進捗状況を取得
 */
router.get('/', (req, res) => {
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  // プロジェクトルートからの絶対パス構築
  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);

  if (!checkExists(learningSourcePath)) {
    return res.status(404).json({
      error: 'Learning source directory not found',
      path: learningSourcePath,
    });
  }

  const progress = {};

  for (const [key, dataset] of Object.entries(DATASETS)) {
    progress[key] = getDatasetProgress(learningSourcePath, key, dataset);
  }

  // 全体統計
  const allSteps = Object.values(progress).reduce(
    (acc, ds) => acc + ds.progress.total,
    0
  );
  const completedAllSteps = Object.values(progress).reduce(
    (acc, ds) => acc + ds.progress.completed,
    0
  );
  const overallPercent = Math.round((completedAllSteps / allSteps) * 100);

  res.json({
    learningSourceDir: learningSourceDir,
    datasets: progress,
    overall: {
      completed: completedAllSteps,
      total: allSteps,
      percent: overallPercent,
      completedDatasets: Object.values(progress).filter(
        (ds) => ds.status === 'completed'
      ).length,
      totalDatasets: Object.keys(progress).length,
    },
  });
});

/**
 * GET /api/dataset-progress/:datasetKey
 * 特定のデータセットの詳細な進捗情報を取得
 */
router.get('/:datasetKey', (req, res) => {
  const { datasetKey } = req.params;
  const learningSourceDir = process.env.LEARNING_SOURCE_DIR;

  if (!learningSourceDir) {
    return res.status(500).json({
      error: 'LEARNING_SOURCE_DIR environment variable is not set',
    });
  }

  const dataset = DATASETS[datasetKey];
  if (!dataset) {
    return res.status(404).json({
      error: 'Dataset not found',
      availableDatasets: Object.keys(DATASETS),
    });
  }

  const projectRoot = path.resolve(__dirname, '..', '..');
  const learningSourcePath = path.join(projectRoot, learningSourceDir);

  if (!checkExists(learningSourcePath)) {
    return res.status(404).json({
      error: 'Learning source directory not found',
      path: learningSourcePath,
    });
  }

  const progress = getDatasetProgress(learningSourcePath, datasetKey, dataset);

  res.json(progress);
});

module.exports = router;
