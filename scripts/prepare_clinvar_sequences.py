#!/usr/bin/env python3
"""
Prepare ClinVar benchmark sequences from HuggingFace dataset and GRCh38 reference.

Usage:
  python prepare_clinvar_sequences.py \
      --ref_fasta GCA_000001405.28_GRCh38.p13_genomic.fna.gz \
      --output_file clinvar_sequences.csv \
      --flank 64 \
      --max_samples 100
"""

import argparse
import re
from datasets import load_dataset
from pyfaidx import Fasta
import pandas as pd

def build_chrom_mapping(ref_genome):
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

def main():
    parser = argparse.ArgumentParser(description="Prepare ClinVar benchmark sequences")
    parser.add_argument("--ref_fasta", type=str, required=True,
                        help="Path to GRCh38 genomic FASTA (e.g. GCA_000001405.28_GRCh38.p13_genomic.fna.gz)")
    parser.add_argument("--output_file", type=str, default="clinvar_sequences.csv",
                        help="Output CSV file")
    parser.add_argument("--flank", type=int, default=64,
                        help="Number of bp to take on each side (default=64 → 128bp window)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to process (default: all)")

    args = parser.parse_args()

    dataset = load_dataset("songlab/clinvar")
    df = dataset["test"].to_pandas()

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Processing only first {args.max_samples} variants for testing")

    ref_genome = Fasta(args.ref_fasta)
    mapping = build_chrom_mapping(ref_genome)

    records = []
    for i, row in df.iterrows():
        try:
            ref_seq, var_seq = get_sequences(
                ref_genome, mapping,
                row["chrom"], row["pos"],
                row["ref"], row["alt"],
                flank=args.flank
            )
            records.append({
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"],
                "label": row["label"],
                "reference_sequence": ref_seq,
                "variant_sequence": var_seq
            })
        except Exception as e:
            print(f"Error at {row['chrom']}:{row['pos']} - {e}")
            continue

    df_out = pd.DataFrame(records)
    df_out.to_csv(args.output_file, index=False)
    print(f"Saved {len(df_out)} variants with sequences → {args.output_file}")

if __name__ == "__main__":
    main()
