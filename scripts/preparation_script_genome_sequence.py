from argparse import ArgumentParser
from pathlib import Path
import logging

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from genome_sequence.dataset.refseq.download_refseq import download_refseq
from genome_sequence.dataset.refseq.fasta_to_raw import fasta_to_raw_genome

# from genome_sequence.dataset.train_tokenizer import train_tokenizer
from genome_sequence.dataset.sentence_piece_tokenizer import train_tokenizer
from genome_sequence.dataset.tokenizer import raw_to_parquet
from genome_sequence.utils.config import GenomeSequenceConfig
from core.base import setup_logging

from config.paths import GENOME_SEQUENCE_DIR

logger = logging.getLogger(__name__)


def create_distribution_plot(data):
    plt.hist(data["train"]["num_tokens"], bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths")
    plt.savefig("assets/img/genome_sequence_tokenized_lengths_dist.png")
    plt.close()
    logger.info(msg="Saved distribution of tokenized dataset lengths to assets/img/genome_sequence_tokenized_lengths_dist.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation
    setup_logging(GENOME_SEQUENCE_DIR)

    # 各処理段階のマーカーファイルパス
    download_marker = Path(GENOME_SEQUENCE_DIR) / "download_complete.marker"
    fasta_to_raw_marker = Path(GENOME_SEQUENCE_DIR) / "fasta_to_raw_complete.marker"
    train_tokenizer_marker = Path(GENOME_SEQUENCE_DIR) / "train_tokenizer_complete.marker"
    raw_to_parquet_marker = Path(GENOME_SEQUENCE_DIR) / "raw_to_parquet_complete.marker"
    
    # 出力ディレクトリとファイルの存在確認用パス
    raw_files_dir = Path(GENOME_SEQUENCE_DIR) / "raw_files"
    tokenizer_model = Path(GENOME_SEQUENCE_DIR) / "spm_tokenizer.model"
    parquet_dir = Path(GENOME_SEQUENCE_DIR) / "parquet_files"

    # 進捗状況の確認
    logger.info("=== Genome Sequence Dataset Preparation Progress ===")
    steps_completed = 0
    total_steps = 4
    
    if download_marker.exists():
        logger.info("✓ Step 1/4: RefSeq download - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 1/4: RefSeq download - PENDING")
        
    if fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("✓ Step 2/4: FASTA to raw conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 2/4: FASTA to raw conversion - PENDING")
        
    if train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("✓ Step 3/4: Tokenizer training - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 3/4: Tokenizer training - PENDING")
        
    if raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("✓ Step 4/4: Raw to Parquet conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 4/4: Raw to Parquet conversion - PENDING")
        
    logger.info(f"Progress: {steps_completed}/{total_steps} steps completed")
    
    if steps_completed == total_steps and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        
    logger.info("====================================================")

    # process1
    if not args.force and download_marker.exists():
        logger.info("👉Process1 : RefSeq dataset download already completed. Skipping...")
        logger.info("Use --force option to re-download.")
    else:
        if args.force:
            logger.info("👉Process1 : Force option specified. Re-downloading RefSeq dataset...")
        else:
            logger.info("👉Process1 : Downloading RefSeq dataset...")
        download_refseq(GENOME_SEQUENCE_DIR, cfg.path_species, cfg.num_worker)
        download_marker.touch()
        logger.info("RefSeq download completed.")

    # process2
    if not args.force and fasta_to_raw_marker.exists() and raw_files_dir.exists() and any(raw_files_dir.glob("*.raw")):
        logger.info("👉Process2 : FASTA to raw conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
    else:
        if args.force:
            logger.info("👉Process2 : Force option specified. Reconverting FASTA to raw text...")
        else:
            logger.info("👉Process2 : Converting FASTA to raw text...")
        logger.info(f" - Base Directory : {GENOME_SEQUENCE_DIR}")
        logger.info(f" - Number of Workers : {cfg.num_worker}")
        logger.info(f" - Max Lines per File : {cfg.max_lines_per_file}")
        fasta_to_raw_genome(GENOME_SEQUENCE_DIR, cfg.num_worker, cfg.max_lines_per_file)
        fasta_to_raw_marker.touch()
        logger.info("FASTA to raw conversion completed.")

    # process3
    if not args.force and train_tokenizer_marker.exists() and tokenizer_model.exists():
        logger.info("👉Process3 : Tokenizer training already completed. Skipping...")
        logger.info("Use --force option to retrain tokenizer.")
    else:
        if args.force:
            logger.info("👉Process3 : Force option specified. Retraining tokenizer...")
        else:
            logger.info("👉Process3 : Training tokenizer...")
        logger.info(f" - Base Directory : {GENOME_SEQUENCE_DIR}")
        logger.info(f" - vocab size : {cfg.vocab_size}")
        logger.info(f" - max lines per file : {cfg.max_lines_per_file}")
        logger.info(f" - input sentence size : {cfg.input_sentence_size}")
        train_tokenizer(GENOME_SEQUENCE_DIR, cfg.vocab_size, cfg.max_lines_per_file, cfg.input_sentence_size)
        train_tokenizer_marker.touch()
        logger.info("Tokenizer training completed.")

    # process4
    if not args.force and raw_to_parquet_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("👉Process4 : Raw to Parquet conversion already completed. Skipping...")
        logger.info("Use --force option to reconvert.")
    else:
        if args.force:
            logger.info("👉Process4 : Force option specified. Reconverting raw text to Parquet...")
        else:
            logger.info("👉Process4 : Converting raw text to Parquet...")
        raw_to_parquet(GENOME_SEQUENCE_DIR)
        raw_to_parquet_marker.touch()
        logger.info("Raw to Parquet conversion completed.")

    # process5
    logger.info("👉Process5 : Loading Parquet dataset and generating statistics...")
    try:
        data = load_dataset(
            "parquet",
            data_files=[str(Path(GENOME_SEQUENCE_DIR) / "parquet_files")],
            cache_dir=str(Path(GENOME_SEQUENCE_DIR) / "hf_cache"),
        )

        logger.info("👍Complete.")
        logger.info(f"Number of sequence: {len(data['train'])}")
        logger.info(f"Size of the vocabulary: {cfg.vocab_size}")
        logger.info(f"Number of tokens: {sum(data['train']['num_tokens'])}")
        
        # 分布プロットの生成（forceオプションまたはプロットが存在しない場合のみ）
        plot_file = Path("assets/img/genome_sequence_tokenized_lengths_dist.png")
        if args.force or not plot_file.exists():
            if args.force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")
            create_distribution_plot(data)
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")
            logger.info("Use --force option to regenerate plot.")
            
        logger.info("Genome sequence dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        logger.error("Some processing steps may have failed. Please check the logs and consider using --force option.")
        exit(1)
