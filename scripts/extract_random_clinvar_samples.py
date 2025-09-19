#!/usr/bin/env python3
"""
ClinVarデータからランダムサンプリングを行うスクリプト

使用例:
1. 既存CSVファイルからランダム抽出:
   python extract_random_clinvar_samples.py --input_csv clinvar_sequences.csv --output_csv random_2000.csv --num_samples 2000

2. データセットから直接ランダム抽出して配列生成:
   python extract_random_clinvar_samples.py --ref_fasta GCA_000001405.28_GRCh38.p13_genomic.fna --output_csv random_2000.csv --num_samples 2000 --flank 64
"""

import argparse
import pandas as pd
import numpy as np
import os
import gzip
import shutil
import re
from datasets import load_dataset
from pyfaidx import Fasta

def build_chrom_mapping(ref_genome):
    """染色体マッピングを構築"""
    headers = [ref_genome[seq].long_name for seq in ref_genome.keys()]
    mapping = {}
    for h in headers:
        m = re.search(r"^(CM\d+\.\d+).*chromosome (\w+)", h)
        if m:
            seq_id = m.group(1)
            chrom = m.group(2)
            if chrom.lower().startswith("mito"):
                chrom = "MT"
            mapping[chrom] = seq_id
    return mapping

def get_sequences(ref_genome, mapping, chrom, pos, ref, alt, flank=64):
    """参照配列と変異配列を取得"""
    seq_id = mapping[str(chrom)]
    start = pos - flank
    end = pos + flank
    ref_seq = ref_genome[seq_id][start-1:end].seq.upper()

    center_base = ref_seq[flank]
    if center_base != ref.upper():
        print(f"Warning: reference mismatch at {chrom}:{pos}, expected {ref}, got {center_base}")

    seq_list = list(ref_seq)
    seq_list[flank] = alt.upper()
    var_seq = "".join(seq_list)

    return ref_seq, var_seq

def extract_from_csv(input_csv, output_csv, num_samples, seed=42):
    """既存のCSVファイルからランダムサンプリング"""
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original dataset size: {len(df)}")
    
    if len(df) < num_samples:
        print(f"Warning: Dataset has only {len(df)} samples, less than requested {num_samples}")
        num_samples = len(df)
    
    # ランダムサンプリング
    np.random.seed(seed)
    sampled_df = df.sample(n=num_samples, random_state=seed)
    
    # Clinical Significanceの分布を確認
    if 'ClinicalSignificance' in sampled_df.columns:
        print("\nClinical Significance distribution in sample:")
        clin_sig_counts = sampled_df['ClinicalSignificance'].value_counts()
        for sig, count in clin_sig_counts.items():
            print(f"  {sig}: {count}")
    
    # 保存
    sampled_df.to_csv(output_csv, index=False)
    print(f"Saved {len(sampled_df)} random samples to {output_csv}")
    
    return sampled_df

def extract_from_dataset(ref_fasta, output_csv, num_samples, flank=64, seed=42):
    """データセットから直接ランダムサンプリングして配列生成"""
    print("Loading ClinVar dataset from HuggingFace...")
    dataset = load_dataset("gonzalobenegas/clinvar")
    df = dataset["test"].to_pandas()
    
    print(f"Total dataset size: {len(df)}")
    
    if len(df) < num_samples:
        print(f"Warning: Dataset has only {len(df)} samples, less than requested {num_samples}")
        num_samples = len(df)
    
    # ランダムサンプリング
    np.random.seed(seed)
    sampled_df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
    
    print(f"Sampled {len(sampled_df)} variants")
    
    # Clinical Significanceの分布を確認
    if 'clin_sig' in sampled_df.columns:
        print("\nClinical Significance distribution in sample:")
        clin_sig_counts = sampled_df['clin_sig'].value_counts()
        for sig, count in clin_sig_counts.items():
            print(f"  {sig}: {count}")
    
    # ファイル形式の自動対応
    ref_fasta_path = ref_fasta
    if ref_fasta_path.endswith('.gz'):
        uncompressed_path = ref_fasta_path.replace('.gz', '')
        if os.path.exists(uncompressed_path):
            print(f"Using uncompressed FASTA file: {uncompressed_path}")
            ref_fasta_path = uncompressed_path
        else:
            print(f"Uncompressing {ref_fasta_path} for compatibility...")
            with gzip.open(ref_fasta_path, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            ref_fasta_path = uncompressed_path
            print(f"Uncompressed to: {ref_fasta_path}")
    
    # 参照ゲノムの読み込み
    print("Loading reference genome...")
    ref_genome = Fasta(ref_fasta_path)
    mapping = build_chrom_mapping(ref_genome)
    
    # 配列生成
    print("Generating sequences...")
    records = []
    for i, row in sampled_df.iterrows():
        try:
            ref_seq, var_seq = get_sequences(
                ref_genome, mapping,
                row["chrom"], row["pos"],
                row["ref"], row["alt"],
                flank=flank
            )
            
            record = {
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"],
                "reference_sequence": ref_seq,
                "variant_sequence": var_seq
            }
            
            # ClinicalSignificanceを追加
            if "clin_sig" in row:
                record["ClinicalSignificance"] = row["clin_sig"]
            else:
                record["ClinicalSignificance"] = None
            
            records.append(record)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(sampled_df)} variants")
                
        except Exception as e:
            print(f"Error at {row['chrom']}:{row['pos']} - {e}")
            continue
    
    # CSVに保存
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    
    print(f"Saved {len(df_out)} variants with sequences to {output_csv}")
    
    # 最終統計
    if 'ClinicalSignificance' in df_out.columns:
        print("\nFinal Clinical Significance distribution:")
        final_counts = df_out['ClinicalSignificance'].value_counts()
        for sig, count in final_counts.items():
            print(f"  {sig}: {count}")
    
    return df_out

def main():
    parser = argparse.ArgumentParser(description="Extract random samples from ClinVar data")
    parser.add_argument("--input_csv", type=str, 
                       help="Input CSV file (for extraction from existing file)")
    parser.add_argument("--ref_fasta", type=str,
                       help="Reference FASTA file (for extraction from dataset)")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file")
    parser.add_argument("--num_samples", type=int, default=2000,
                       help="Number of samples to extract (default: 2000)")
    parser.add_argument("--flank", type=int, default=64,
                       help="Flank size for sequence extraction (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # 入力モードの確認
    if args.input_csv and args.ref_fasta:
        print("Error: Specify either --input_csv OR --ref_fasta, not both")
        return
    elif args.input_csv:
        print("Mode: Extracting from existing CSV file")
        extract_from_csv(args.input_csv, args.output_csv, args.num_samples, args.seed)
    elif args.ref_fasta:
        print("Mode: Extracting from dataset with sequence generation")
        extract_from_dataset(args.ref_fasta, args.output_csv, args.num_samples, args.flank, args.seed)
    else:
        print("Error: Specify either --input_csv or --ref_fasta")
        parser.print_help()

if __name__ == "__main__":
    main()
