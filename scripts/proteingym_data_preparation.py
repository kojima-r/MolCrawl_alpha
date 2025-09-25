#!/usr/bin/env python3
"""
ProteinGymデータセットのダウンロードと前処理ユーティリティ

このスクリプトは、ProteinGymデータセットをダウンロードし、
評価用に適切な形式に前処理します。
"""

import os
import argparse
import requests
import pandas as pd
import numpy as np
import zipfile
import logging
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProteinGymDataDownloader:
    """ProteinGymデータセットのダウンロードと前処理クラス"""
    
    # ProteinGym v1.3データセットの公式URL
    PROTEINGYM_URLS = {
        # DMS (Deep Mutational Scanning) データ - メイン評価用
        'substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip',
        'indels': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip',
        
        # 参照ファイル - アッセイメタデータ
        'reference_substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv',
        'reference_indels': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_indels.csv',
        
        # 臨床変異データ - 補完評価用
        'clinical_substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_substitutions.zip',
        'clinical_indels': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_indels.zip',
        'clinical_reference_substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_substitutions.csv',
        'clinical_reference_indels': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_indels.csv',
        
        # 生データ（必要に応じて）
        'raw_substitutions': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/substitutions_raw_DMS.zip',
        'raw_indels': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/indels_raw_DMS.zip',
        
        # 多重配列アライメント（高度な分析用）
        'msa_dms': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip',
        'msa_clinical': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_msa_files.zip',
        
        # タンパク質構造（構造ベース分析用）
        'structures': 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ProteinGym_AF2_structures.zip'
    }
    
    def __init__(self, data_dir='./proteingym_data'):
        """
        初期化
        
        Args:
            data_dir (str): データ保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, filename=None, force_download=False):
        """
        ファイルをダウンロード
        
        Args:
            url (str): ダウンロードURL
            filename (str): 保存ファイル名（Noneの場合はURLから推定）
            force_download (bool): 既存ファイルを上書きするか
            
        Returns:
            str: ダウンロードされたファイルのパス
        """
        if filename is None:
            filename = os.path.basename(urlparse(url).path)
        
        filepath = self.data_dir / filename
        
        if filepath.exists() and not force_download:
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        logger.info(f"Downloading {url} to {filepath}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded successfully: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def extract_zip(self, zip_path, extract_dir=None):
        """
        ZIPファイルを展開
        
        Args:
            zip_path (str): ZIPファイルのパス
            extract_dir (str): 展開先ディレクトリ（Noneの場合は同じディレクトリ）
            
        Returns:
            str: 展開先ディレクトリ
        """
        zip_path = Path(zip_path)
        
        if extract_dir is None:
            extract_dir = zip_path.parent / zip_path.stem
        else:
            extract_dir = Path(extract_dir)
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted successfully to {extract_dir}")
        return str(extract_dir)
    
    def download_proteingym_data(self, data_type='substitutions', force_download=False):
        """
        ProteinGymデータセットをダウンロード
        
        Args:
            data_type (str): データタイプ
                          - 'substitutions': DMS単一置換データ（推奨）
                          - 'indels': DMS挿入・欠失データ
                          - 'clinical_substitutions': 臨床変異置換データ
                          - 'clinical_indels': 臨床変異挿入・欠失データ
                          - 'reference_substitutions': DMS置換参照ファイル
                          - 'reference_indels': DMS挿入・欠失参照ファイル
                          - 'msa_dms': 多重配列アライメント（DMS）
                          - 'structures': タンパク質構造データ
            force_download (bool): 強制ダウンロード
            
        Returns:
            str: ダウンロードされたファイル/ディレクトリのパス
        """
        if data_type not in self.PROTEINGYM_URLS:
            available_types = list(self.PROTEINGYM_URLS.keys())
            raise ValueError(f"Invalid data_type: {data_type}. Choose from {available_types}")
        
        url = self.PROTEINGYM_URLS[data_type]
        downloaded_file = self.download_file(url, force_download=force_download)
        
        # ZIPファイルの場合は展開
        if downloaded_file.endswith('.zip'):
            extracted_dir = self.extract_zip(downloaded_file)
            return extracted_dir
        else:
            return downloaded_file
    
    def load_reference_file(self, reference_path=None, data_type='substitutions'):
        """
        ProteinGym参照ファイルを読み込み
        
        Args:
            reference_path (str): 参照ファイルのパス（Noneの場合は自動ダウンロード）
            data_type (str): データタイプ ('substitutions' or 'indels')
            
        Returns:
            pd.DataFrame: 参照データ
        """
        if reference_path is None:
            reference_key = f'reference_{data_type}'
            if reference_key not in self.PROTEINGYM_URLS:
                reference_key = 'reference_substitutions'  # デフォルト
            reference_path = self.download_proteingym_data(reference_key)
        
        logger.info(f"Loading reference file: {reference_path}")
        df = pd.read_csv(reference_path)
        logger.info(f"Loaded {len(df)} assays from reference file")
        logger.info(f"Available columns: {list(df.columns)}")
        
        return df
    
    def prepare_evaluation_data(self, assay_id, data_dir=None, max_variants=None):
        """
        特定のアッセイの評価データを準備
        
        Args:
            assay_id (str): アッセイID
            data_dir (str): データディレクトリ（Noneの場合は自動ダウンロード）
            max_variants (int): 最大変異数（制限しない場合はNone）
            
        Returns:
            pd.DataFrame: 評価用データ
        """
        if data_dir is None:
            data_dir = self.download_proteingym_data('substitutions')
        
        # アッセイファイルを探す
        assay_file = None
        data_path = Path(data_dir)
        
        for file_path in data_path.rglob(f"{assay_id}.csv"):
            assay_file = file_path
            break
        
        if assay_file is None:
            raise FileNotFoundError(f"Assay file not found for ID: {assay_id}")
        
        logger.info(f"Loading assay data: {assay_file}")
        df = pd.read_csv(assay_file)
        
        # 必要なカラムをチェック
        required_columns = ['mutant', 'mutated_sequence', 'DMS_score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # データのクリーニング
        df = df.dropna(subset=['DMS_score'])
        
        # 最大変異数の制限
        if max_variants and len(df) > max_variants:
            logger.info(f"Limiting to {max_variants} variants (from {len(df)})")
            df = df.sample(n=max_variants, random_state=42)
        
        logger.info(f"Prepared {len(df)} variants for evaluation")
        logger.info(f"DMS score range: {df['DMS_score'].min():.3f} to {df['DMS_score'].max():.3f}")
        
        return df
    
    def create_test_dataset(self, output_file, n_variants=100):
        """
        テスト用の小さなデータセットを作成
        
        Args:
            output_file (str): 出力ファイルパス
            n_variants (int): 変異数
        """
        logger.info(f"Creating test dataset with {n_variants} variants")
        
        # サンプルタンパク質配列
        base_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAAFRVTLNEKLATWTEESS"
        
        # ランダムな変異を生成
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        data = []
        
        np.random.seed(42)
        
        for i in range(n_variants):
            if i == 0:
                # 野生型
                mutant = 'WT'
                mutated_seq = base_sequence
                score = 1.0
            else:
                # ランダム変異
                pos = np.random.randint(1, len(base_sequence) + 1)
                orig_aa = base_sequence[pos - 1]
                mut_aa = np.random.choice([aa for aa in amino_acids if aa != orig_aa])
                
                mutant = f"{orig_aa}{pos}{mut_aa}"
                mutated_seq = base_sequence[:pos-1] + mut_aa + base_sequence[pos:]
                
                # ランダムなスコア（より現実的な分布）
                score = np.random.beta(2, 5)  # 0に偏った分布
            
            data.append({
                'mutant': mutant,
                'mutated_sequence': mutated_seq,
                'DMS_score': score,
                'protein_name': 'TEST_PROTEIN'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Test dataset created: {output_file}")
        logger.info(f"Score statistics: mean={df['DMS_score'].mean():.3f}, std={df['DMS_score'].std():.3f}")
    
    def download_recommended_datasets(self, force_download=False):
        """
        protein_sequence評価に推奨されるデータセットをダウンロード
        
        Args:
            force_download (bool): 強制ダウンロード
            
        Returns:
            dict: ダウンロードされたファイルパスの辞書
        """
        logger.info("Downloading recommended datasets for protein_sequence evaluation...")
        
        recommended_datasets = [
            'substitutions',  # メイン評価用：DMS単一置換データ
            'reference_substitutions',  # アッセイメタデータ
            'clinical_substitutions',  # 補完評価用：臨床変異データ
            'clinical_reference_substitutions'  # 臨床変異メタデータ
        ]
        
        downloaded_paths = {}
        
        for dataset in recommended_datasets:
            try:
                logger.info(f"Downloading {dataset}...")
                path = self.download_proteingym_data(dataset, force_download=force_download)
                downloaded_paths[dataset] = path
                logger.info(f"✓ {dataset} downloaded to: {path}")
            except Exception as e:
                logger.warning(f"Failed to download {dataset}: {e}")
                downloaded_paths[dataset] = None
        
        return downloaded_paths
    
    def get_small_test_assays(self, reference_df, max_assays=5, max_variants_per_assay=500):
        """
        テスト用の小さなアッセイを選択
        
        Args:
            reference_df (pd.DataFrame): 参照データフレーム
            max_assays (int): 最大アッセイ数
            max_variants_per_assay (int): アッセイあたりの最大変異数
            
        Returns:
            list: 選択されたアッセイIDのリスト
        """
        # 変異数でフィルタリング
        if 'DMS_total_number_mutants' in reference_df.columns:
            filtered_df = reference_df[
                (reference_df['DMS_total_number_mutants'] <= max_variants_per_assay) &
                (reference_df['DMS_total_number_mutants'] >= 50)  # 最小50変異
            ]
        else:
            filtered_df = reference_df
        
        # ランダムに選択
        if len(filtered_df) > max_assays:
            selected_df = filtered_df.sample(n=max_assays, random_state=42)
        else:
            selected_df = filtered_df
        
        assay_ids = selected_df['DMS_id'].tolist()
        
        logger.info(f"Selected {len(assay_ids)} test assays:")
        for assay_id in assay_ids:
            row = selected_df[selected_df['DMS_id'] == assay_id].iloc[0]
            n_variants = row.get('DMS_total_number_mutants', 'Unknown')
            protein_name = row.get('UniProt_ID', 'Unknown')
            logger.info(f"  - {assay_id}: {protein_name} ({n_variants} variants)")
        
        return assay_ids

def main():
    parser = argparse.ArgumentParser(description='ProteinGym data downloader and preparation utility')
    parser.add_argument('--data_dir', type=str, default='./proteingym_data',
                       help='Data directory for ProteinGym datasets')
    parser.add_argument('--download', 
                       choices=['substitutions', 'indels', 'clinical_substitutions', 'clinical_indels',
                               'reference_substitutions', 'reference_indels', 'msa_dms', 'structures', 
                               'recommended', 'all'],
                       help='Download ProteinGym data. "recommended" downloads essential datasets for protein_sequence evaluation')
    parser.add_argument('--prepare_assay', type=str,
                       help='Prepare evaluation data for specific assay ID')
    parser.add_argument('--max_variants', type=int,
                       help='Maximum number of variants to include')
    parser.add_argument('--create_test', type=str,
                       help='Create test dataset (provide output filename)')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Size of test dataset')
    parser.add_argument('--force', action='store_true',
                       help='Force download even if files exist')
    parser.add_argument('--list_assays', action='store_true',
                       help='List available assays from reference file')
    parser.add_argument('--get_test_assays', type=int, default=5,
                       help='Get small test assays (specify number of assays)')
    parser.add_argument('--data_type', choices=['substitutions', 'indels'], default='substitutions',
                       help='Data type for reference file loading')
    
    args = parser.parse_args()
    
    downloader = ProteinGymDataDownloader(data_dir=args.data_dir)
    
    try:
        # ダウンロード
        if args.download:
            if args.download == 'recommended':
                # protein_sequence評価に推奨されるデータセットをダウンロード
                downloaded_paths = downloader.download_recommended_datasets(force_download=args.force)
                logger.info("Recommended datasets downloaded:")
                for dataset, path in downloaded_paths.items():
                    if path:
                        logger.info(f"  ✓ {dataset}: {path}")
                    else:
                        logger.warning(f"  ✗ {dataset}: Failed to download")
            elif args.download == 'all':
                # 主要なデータセットをすべてダウンロード
                main_datasets = ['substitutions', 'indels', 'reference_substitutions', 'reference_indels',
                               'clinical_substitutions', 'clinical_indels']
                for data_type in main_datasets:
                    try:
                        downloader.download_proteingym_data(data_type, force_download=args.force)
                    except Exception as e:
                        logger.warning(f"Failed to download {data_type}: {e}")
            else:
                downloader.download_proteingym_data(args.download, force_download=args.force)
        
        # アッセイリスト表示
        if args.list_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            print(f"\nAvailable {args.data_type} assays:")
            for _, row in ref_df.iterrows():
                protein_id = row.get('UniProt_ID', row.get('protein_name', 'N/A'))
                n_variants = row.get('DMS_total_number_mutants', 'N/A')
                organism = row.get('taxon', 'N/A')
                print(f"  {row['DMS_id']}: {protein_id} ({organism}) - {n_variants} variants")
            
            print(f"\nTotal assays: {len(ref_df)}")
        
        # テスト用アッセイの取得
        if args.get_test_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            test_assays = downloader.get_small_test_assays(ref_df, max_assays=args.get_test_assays)
            
            print(f"\nRecommended test assays for quick evaluation:")
            for assay_id in test_assays:
                print(f"  {assay_id}")
            
            # サンプル実行コマンドを表示
            if test_assays:
                print(f"\nExample evaluation command:")
                print(f"python scripts/proteingym_evaluation.py \\")
                print(f"    --model_path gpt2-output/protein_sequence-small/ckpt.pt \\")
                print(f"    --proteingym_data proteingym_data/{test_assays[0]}.csv \\")
                print(f"    --output_dir results_{test_assays[0]}/")
                print(f"    --batch_size 16")
        
        # アッセイデータ準備
        if args.prepare_assay:
            eval_data = downloader.prepare_evaluation_data(
                args.prepare_assay,
                max_variants=args.max_variants
            )
            
            output_file = f"{args.prepare_assay}_evaluation_data.csv"
            eval_data.to_csv(output_file, index=False)
            logger.info(f"Evaluation data saved to: {output_file}")
        
        # テストデータセット作成
        if args.create_test:
            downloader.create_test_dataset(args.create_test, n_variants=args.test_size)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())