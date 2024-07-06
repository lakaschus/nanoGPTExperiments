# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 1
dataset_id = "stas/openwebtext-10k"
dataset = load_dataset(dataset_id)
dataset_dir = dataset_id.split("/")[-1].replace("-", "_")

file_dir = os.path.dirname(__file__)

if not os.path.exists(file_dir + "/" + dataset_dir):
    os.makedirs(file_dir + "/" + dataset_dir)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(
    test_size=0.1, seed=2351, shuffle=True
)
split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids, "len": len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=["text"],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset["len"])
    filename = os.path.join(os.path.dirname(__file__), dataset_dir, f"{split}.bin")
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example["len"]] = example["ids"]
        idx += example["len"]
    arr.flush()
