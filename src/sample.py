import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sample from a trained GPT model")
    parser.add_argument(
        "--init_from",
        type=str,
        default="resume",
        help="Either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Output directory (ignored if init_from is not 'resume')",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=" ",
        help="Start text.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Number of tokens generated in each sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Retain only the top_k most likely tokens, clamp others to have 0 probability",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for computations",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use PyTorch 2.0 to compile the model to be faster",
    )
    return parser.parse_args()


def main(args):
    # Set up the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Load the model
    if args.init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif args.init_from.startswith("gpt2"):
        model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    # Set up encoding/decoding
    load_meta = False
    if (
        args.init_from == "resume"
        and "config" in checkpoint
        and "dataset" in checkpoint["config"]
    ):
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    start = args.start
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    # Generate samples
    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(
                    x,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                sample = decode(y[0].tolist())
                print(sample)
                with open(f"{args.out_dir}/sample.txt", "a") as f:
                    f.write(f"Sample {k}:\n")
                    f.write(sample)
                    f.write("\n\n")
                print("---------------")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
