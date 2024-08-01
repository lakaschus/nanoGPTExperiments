import os
import numpy as np
from tqdm import tqdm
import torch


class ArithmeticTokenizer:
    def __init__(self):
        self.vocab = {str(i): i for i in range(10)}  # Digits 0-9
        self.vocab["+"] = 10
        self.vocab["="] = 11
        self.vocab["_"] = 14  # problem not yet complete token
        self.vocab["PAD"] = 13  # Padding token
        self.vocab["<pad>"] = 13  # Padding token
        self.vocab["\n"] = 12
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.split_token_id = 12

    def encode(self, text):
        return [self.vocab[char] for char in text]

    def decode(self, tokens):
        toks = tokens
        if isinstance(tokens, torch.Tensor):
            toks = toks.cpu().numpy().tolist()
        return "".join(self.inverse_vocab[token] for token in toks)

    def get_vocab(self):
        return self.vocab


# Generate all possible additions up to max_number
def generate_additions(max_number):
    additions = []
    for i in range(max_number + 1):
        for j in range(max_number + 1 - i):
            result = i + j
            if result <= max_number:
                additions.append(f"{i}+{j}={result}\n")
    return additions


# Process the dataset
def process_dataset(additions):
    tokenized_data = []

    for addition in tqdm(additions, desc="Tokenizing additions"):
        tokenized_data.extend(ArithmeticTokenizer().encode(addition))
    return np.array(tokenized_data, dtype=np.uint16)


# Save dataset to a binary file
def save_dataset(data, filename):
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(len(data),))
    arr[:] = data[:]
    arr.flush()


# Main function
def main():
    max_number = 1000
    output_dir = "./src/data/arithmetic_dataset"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating additions up to {max_number}...")
    additions = generate_additions(max_number)
    # shuffle data
    # np.random.shuffle(additions)
    print(f"Total number of additions: {len(additions)}")

    # sort additions by length
    additions.sort(key=len)

    print("Processing dataset...")
    tokenized_data = process_dataset(additions)

    # Split the data into training and validation sets
    split_index = int(0.9 * len(tokenized_data))  # 90% for training, 10% for validation
    # For each order of magnitude, choose 90% for training and 10% for validation
    # TODO

    train_data = tokenized_data[:split_index]
    val_data = tokenized_data[split_index:]

    print("Saving datasets...")
    save_dataset(train_data, os.path.join(output_dir, "train.bin"))
    save_dataset(val_data, os.path.join(output_dir, "val.bin"))

    print("Datasets saved in the 'arithmetic_dataset' directory")

    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Total tokens: {len(tokenized_data)}")
    print(f"Training tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    print(f"Unique tokens: {len(np.unique(tokenized_data))}")
    print(f"Total dataset size: {tokenized_data.nbytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
