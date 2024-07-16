import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess OpenWebText dataset for training"
    )
    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of workers in .map() call"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="openwebtext-10k",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Size of the test split"
    )
    parser.add_argument(
        "--seed", type=int, default=2351, help="Random seed for dataset splitting"
    )
    parser.add_argument(
        "--encoding", type=str, default="gpt2", help="Tiktoken encoding to use"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload and reprocessing of data"
    )
    return parser.parse_args()


def process(example, enc):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


def check_data_files(output_dir):
    required_files = ["train.bin", "val.bin"]
    return all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)


def main(args):
    # Prepare directories
    dataset_dir = args.dataset_id.split("/")[-1].replace("-", "_")
    file_dir = os.path.dirname(__file__)
    output_dir = os.path.join(file_dir, dataset_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if data files already exist
    if check_data_files(output_dir) and not args.force:
        print(f"Data files already exist in {output_dir}. Use --force to reprocess.")
        return

    # Load and prepare the dataset
    print(f"Loading dataset {args.dataset_id}...")
    dataset = load_dataset(args.dataset_id)

    # Split the dataset
    print("Splitting dataset...")
    split_dataset = dataset["train"].train_test_split(
        test_size=args.test_size, seed=args.seed, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # Initialize the tokenizer
    enc = tiktoken.get_encoding(args.encoding)

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        lambda example: process(example, enc),
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    # Save tokenized datasets
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"])
        filename = os.path.join(output_dir, f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        print(f"Writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example["len"]] = example["ids"]
            idx += example["len"]
        arr.flush()

    print("Data processing complete.")


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
