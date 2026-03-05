#!/usr/bin/env python3
"""
RNA Encoding Validation Script
==============================

Script for detailed encoding integrity verification of RNA datasets
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

from molcrawl.utils.environment_check import check_learning_source_dir

# add project root to path

try:
    import numpy as np
    from datasets import load_dataset
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RNAEncodingValidator:
    """RNA encoding integrity verification class"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.vocab_file = self.dataset_dir / "gene_vocab.json"
        self.parquet_dir = self.dataset_dir / "parquet_files"
        self.cache_dir = self.dataset_dir / "hf_cache"

        self.vocab = None
        self.dataset = None
        self.issues: list[str] = []

    def load_vocabulary(self) -> bool:
        """Load vocabulary file"""
        try:
            if not self.vocab_file.exists():
                self.issues.append(f"Vocabulary file not found: {self.vocab_file}")
                return False

            with open(self.vocab_file, "r") as f:
                self.vocab = json.load(f)

            logger.info(f"✓ Vocabulary loaded: {len(self.vocab)} genes")
            return True

        except Exception as e:
            self.issues.append(f"Failed to load vocabulary: {e}")
            return False

    def load_dataset(self) -> bool:
        """Load parquet dataset"""
        try:
            if not self.parquet_dir.exists():
                self.issues.append(f"Parquet directory not found: {self.parquet_dir}")
                return False

            parquet_files = list(self.parquet_dir.glob("*.parquet"))
            if not parquet_files:
                self.issues.append(f"No parquet files found in: {self.parquet_dir}")
                return False

            logger.info(f"Found {len(parquet_files)} parquet files")

            self.dataset = load_dataset(
                "parquet",
                data_dir=str(self.parquet_dir),
                split="train",
                cache_dir=str(self.cache_dir),
            )

            logger.info(f"✓ Dataset loaded: {len(self.dataset)} sequences")
            return True

        except Exception as e:
            self.issues.append(f"Failed to load dataset: {e}")
            traceback.print_exc()
            return False

    def validate_token_ranges(self) -> bool:
        """Validate token ID range"""
        try:
            if not self.vocab or not self.dataset:
                return False

            vocab_size = len(self.vocab)
            max_vocab_id = max(self.vocab.values()) if self.vocab else -1

            logger.info(f"Vocabulary size: {vocab_size}")
            logger.info(f"Max vocabulary ID: {max_vocab_id}")

            # Check sample token ID
            sample_size = min(1000, len(self.dataset))
            logger.info(f"Checking token ranges in {sample_size} samples...")

            invalid_tokens = []
            out_of_range_count = 0
            total_tokens = 0

            for i in range(sample_size):
                tokens = self.dataset[i]["token"]
                if tokens:  # if not empty
                    total_tokens += len(tokens)
                    for token_id in tokens:
                        if token_id < 0 or token_id > max_vocab_id:
                            invalid_tokens.append((i, token_id))
                            out_of_range_count += 1

            if invalid_tokens:
                self.issues.append(f"Found {out_of_range_count} out-of-range tokens in {len(invalid_tokens)} sequences")
                # show first 10 invalid tokens
                for seq_idx, token_id in invalid_tokens[:10]:
                    self.issues.append(f"  Sequence {seq_idx}: token_id {token_id} (valid range: 0-{max_vocab_id})")
                return False
            else:
                logger.info(f"✓ All {total_tokens} tokens are within valid range (0-{max_vocab_id})")
                return True

        except Exception as e:
            self.issues.append(f"Token range validation failed: {e}")
            return False

    def validate_vocabulary_consistency(self) -> bool:
        """Verify vocabulary consistency"""
        try:
            if not self.vocab:
                return False

            # Duplicate check
            gene_names = list(self.vocab.keys())
            token_ids = list(self.vocab.values())

            duplicate_genes = len(gene_names) != len(set(gene_names))
            duplicate_ids = len(token_ids) != len(set(token_ids))

            if duplicate_genes:
                self.issues.append("Duplicate gene names found in vocabulary")
                return False

            if duplicate_ids:
                self.issues.append("Duplicate token IDs found in vocabulary")
                return False

            # ID range check
            min_id, max_id = min(token_ids), max(token_ids)
            expected_range = list(range(len(token_ids)))
            actual_range = sorted(token_ids)

            if actual_range != expected_range:
                self.issues.append(f"Token ID range is not continuous: expected 0-{len(token_ids) - 1}, got {min_id}-{max_id}")
                missing_ids = set(expected_range) - set(actual_range)
                if missing_ids:
                    self.issues.append(f"Missing token IDs: {sorted(list(missing_ids))[:10]}...")  # First 10
                return False

            logger.info(f"✓ Vocabulary consistency validated: {len(gene_names)} unique genes, IDs 0-{max_id}")
            return True

        except Exception as e:
            self.issues.append(f"Vocabulary consistency validation failed: {e}")
            return False

    def validate_tokenization_quality(self) -> bool:
        """Verify tokenization quality"""
        try:
            if not self.dataset:
                return False

            # Collect statistics
            token_counts = []
            empty_sequences = 0
            total_sequences = min(1000, len(self.dataset))

            for i in range(total_sequences):
                tokens = self.dataset[i]["token"]
                token_count = len(tokens) if tokens else 0

                if token_count == 0:
                    empty_sequences += 1
                else:
                    token_counts.append(token_count)

            if not token_counts:
                self.issues.append("All sequences are empty (no tokens)")
                return False

            # statistical calculation
            avg_tokens = np.mean(token_counts)
            median_tokens = np.median(token_counts)
            min_tokens = np.min(token_counts)
            max_tokens = np.max(token_counts)

            logger.info(f"Tokenization statistics (sample of {total_sequences}):")
            logger.info(f"  Empty sequences: {empty_sequences} ({empty_sequences / total_sequences * 100:.1f}%)")
            logger.info(f"  Average tokens per sequence: {avg_tokens:.1f}")
            logger.info(f"  Median tokens per sequence: {median_tokens:.1f}")
            logger.info(f"  Token count range: {min_tokens}-{max_tokens}")

            # Quality check
            if empty_sequences / total_sequences > 0.5:
                self.issues.append(f"Too many empty sequences: {empty_sequences}/{total_sequences}")
                return False

            if avg_tokens < 10:
                self.issues.append(f"Average tokens per sequence too low: {avg_tokens}")
                return False

            logger.info("✓ Tokenization quality validated")
            return True

        except Exception as e:
            self.issues.append(f"Tokenization quality validation failed: {e}")
            return False

    def check_encoding_pipeline_consistency(self) -> bool:
        """Check consistency across encoding pipeline"""
        try:
            # Consistency check of different tokenization methods
            if not self.dataset or not self.vocab:
                return False

            # Check the difference between tokenization between Geneformer and scgpt
            # (This part needs to be adjusted according to the actual implementation)

            # Check the integrity of the token_count field in the dataset
            inconsistent_counts = 0
            total_checked = min(100, len(self.dataset))

            for i in range(total_checked):
                item = self.dataset[i]
                tokens = item["token"]
                reported_count = item.get("token_count", len(tokens))
                actual_count = len(tokens) if tokens else 0

                if actual_count != reported_count:
                    inconsistent_counts += 1

            if inconsistent_counts > 0:
                self.issues.append(f"Token count inconsistencies found: {inconsistent_counts}/{total_checked}")
                return False

            logger.info("✓ Encoding pipeline consistency validated")
            return True

        except Exception as e:
            self.issues.append(f"Pipeline consistency check failed: {e}")
            return False

    def run_full_validation(self) -> bool:
        """Run full validation"""
        logger.info("=== RNA Encoding Validation Started ===")

        validation_steps = [
            ("Loading vocabulary", self.load_vocabulary),
            ("Loading dataset", self.load_dataset),
            ("Validating vocabulary consistency", self.validate_vocabulary_consistency),
            ("Validating token ranges", self.validate_token_ranges),
            ("Validating tokenization quality", self.validate_tokenization_quality),
            ("Checking pipeline consistency", self.check_encoding_pipeline_consistency),
        ]

        all_passed = True

        for step_name, step_func in validation_steps:
            logger.info(f"Step: {step_name}")
            try:
                if not step_func():
                    logger.error(f"❌ {step_name} FAILED")
                    all_passed = False
                else:
                    logger.info(f"✓ {step_name} PASSED")
            except Exception as e:
                logger.error(f"❌ {step_name} ERROR: {e}")
                all_passed = False

        # Result summary
        logger.info("=== Validation Results ===")
        if all_passed:
            logger.info("✅ All validation checks PASSED")
        else:
            logger.error("❌ Validation FAILED")
            logger.error("Issues found:")
            for issue in self.issues:
                logger.error(f"  - {issue}")

        return all_passed


def main():
    """Main function"""
    import argparse

    # Get LEARNING_SOURCE_DIR
    learning_source_dir = check_learning_source_dir()

    parser = argparse.ArgumentParser(description="RNA Encoding Validation")
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="rna",
        help="Dataset subdirectory relative to LEARNING_SOURCE_DIR (default: rna)",
    )

    args = parser.parse_args()

    # Build dataset directory
    dataset_dir = os.path.join(learning_source_dir, args.dataset_subdir)

    # Check the existence of the directory
    if not os.path.exists(dataset_dir):
        print(f"❌ ERROR: Dataset directory does not exist: {dataset_dir}")
        print(f"Expected structure: {learning_source_dir}/{args.dataset_subdir}")
        print("")
        print("Please verify that:")
        print(f"1. LEARNING_SOURCE_DIR='{learning_source_dir}' is correct")
        print(f"2. The RNA dataset directory exists: {args.dataset_subdir}")
        sys.exit(1)

    validator = RNAEncodingValidator(dataset_dir)
    success = validator.run_full_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
