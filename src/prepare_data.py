import os
import argparse
import logging
import re
from typing import Optional, Union
from Bio import SeqIO
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(input_file_name: str, bidirectional: bool = False) -> list[str]:
    """
    Prepare data from the input fasta file.
    """
    m = re.search(r"PF([0-9]{5})\.fasta", input_file_name)
    if m is None:
        logger.error("Invalid input file name.")
        raise ValueError("Invalid input file name. Input file should be in the format PFXXXXX.fasta. Please consider downloading the data using the download_pfam.py script.")

    prefix = m.group(1)
    seqs = SeqIO.parse(open(input_file_name, "r"), "fasta")
    parsed_seqs = []
    for s in seqs:
        parsed_seqs.append(f"<|pf{prefix}|>1{str(s.seq)}2")
        if bidirectional:
            parsed_seqs.append(f"<|pf{prefix}|>2{str(s.seq)[::-1]}1")
    return parsed_seqs


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    if not 0 <= args.train_split_ratio <= 1:
        raise ValueError("Train-test split ratio must be between 0 and 1.")

    train_data = []
    test_data = []
    for input_file in args.input_files:
        data = prepare_data(input_file, args.bidirectional)
        logging.info(f"Loaded {len(data)} sequences from {input_file}")
        np.random.shuffle(data)
        split_idx = int(len(data) * args.train_split_ratio)
        train_data.extend(data[:split_idx])
        test_data.extend(data[split_idx:])
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    if args.bidirectional:
        logging.info("Data is bidirectional. Each sequence will be stored in both directions.")

    logging.info(f"Train data: {len(train_data)} sequences")
    logging.info(f"Test data: {len(test_data)} sequences")

    logging.info(f"Saving training data to {args.output_file_train}")
    with open(args.output_file_train, "w") as f:
        for line in train_data:
            f.write(line + "\n")

    logging.info(f"Saving test data to {args.output_file_test}")
    with open(args.output_file_test, "w") as f:
        for line in test_data:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True, help="Input fasta files."
    )
    parser.add_argument(
        "--output_file_train", type=str, default="train_data.txt", help="Output file for the train data split. Default: train_data.txt"
    )
    parser.add_argument(
        "--output_file_test", type=str, default="test_data.txt", help="Output file for test data split. Default: test_data.txt"
    )
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        help="Whether to store also the reverse of the sequences. Default: False.",
    )
    parser.add_argument(
        "--train_split_ratio",
        "-s",
        type=float,
        default=0.8,
        help="Train-test split ratio. Default: 0.8",
    )
    parser.add_argument(
        "--seed", type=int, default=69, help="Random seed",
    )
    args = parser.parse_args()
    main(args)
