import os
import time
import math
import argparse
from contextlib import nullcontext
from transformers import GPT2Tokenizer
from torch.nn import functional as F


import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

import mlflow
from mlflow.models.signature import infer_signature


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model")

    # I/O
    parser.add_argument(
        "--out_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=2000, help="Evaluation interval"
    )
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval")
    parser.add_argument(
        "--eval_iters", type=int, default=200, help="Number of evaluation iterations"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If True, script exits right after the first eval",
    )
    parser.add_argument(
        "--always_save_checkpoint",
        action="store_true",
        help="If True, always save a checkpoint after each eval",
        default=False,
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default="scratch",
        choices=["scratch", "resume", "gpt2"],
        help="Where to initialize from",
    )
    parser.add_argument(
        "--num_sample_tokens",
        type=int,
        default=20,
        help="Number of tokens to sample during evaluation",
    )

    # data
    parser.add_argument(
        "--dataset", type=str, default="openwebtext_10k", help="Dataset to use"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="Tokenizer to use"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=5,
        help="Used to simulate larger batch sizes",
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size")

    # model
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--n_head", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--bias", action="store_true", help="Use bias in LayerNorm and Linear layers"
    )

    # optimizer
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=600000,
        help="Maximum number of training iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )

    # learning rate decay
    parser.add_argument(
        "--decay_lr", action="store_true", help="Whether to decay the learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=2000, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=60000,
        help="Number of iterations for LR decay",
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="Minimum learning rate"
    )

    # system
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use PyTorch 2.0 to compile the model",
        default=False,
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="Backend for distributed training"
    )
    parser.add_argument(
        "--azure", action="store_true", help="Use Azure ML datasets", default=False
    )

    # experiment
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="nano-gpt-training",
        help="MLflow experiment name",
    )

    args = parser.parse_args()
    return args


def get_lr(iter_num, args):
    # learning rate decay scheduler (cosine with warmup)
    if iter_num < args.warmup_iters:
        return args.learning_rate * iter_num / args.warmup_iters
    if iter_num > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (iter_num - args.warmup_iters) / (
        args.lr_decay_iters - args.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, data_loader, args, ctx, tokenizer, iter_num):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = data_loader.get_batch(split, mode="eval")
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Sample tokens
    x = torch.zeros((1, 1), dtype=torch.long).to(args.device)
    if args.tokenizer == "arithmetic":
        sample_tokens = [
            0,  # np.random.randint(0, 4),
            tokenizer.vocab["+"],
            1,  # np.random.randint(0, 4),
            tokenizer.vocab["="],
        ]
    else:
        sample_tokens = []
    sample = model.generate(
        torch.tensor(sample_tokens, dtype=torch.long).unsqueeze(0).to(args.device),
        args.num_sample_tokens,
    )
    # Decode sampled tokens
    decoded_sample = tokenizer.decode(sample[0])
    print(f"Sampled text: {decoded_sample}\n")

    # Log sampled text to MLflow
    mlflow.log_text(decoded_sample, f"sampled_text_iter_{iter_num}.txt")

    model.train()
    return out


def load_vocab(type="gpt2"):
    """
    Load the GPT-2 tokenizer and return its vocabulary.
    """
    if type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif type == "arithmetic":
        from data.arithmetic_synthetic import ArithmeticTokenizer

        tokenizer = ArithmeticTokenizer()
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    return vocab, tokenizer


class DataLoader:
    def __init__(self, args, tokenizer, device):
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.train_data, self.val_data = self.get_data()
        self.tokens_per_epoch = len(self.train_data)
        self.training_stage = 0
        self.training_example_id = 0
        self.batch_calls = 0

    def update_training_stage(self):
        # Increment the training stage after 10 calls
        self.batch_calls += 1
        self.training_example_id += 1
        if self.training_example_id > self.training_stage:
            self.training_example_id = 0

        if self.batch_calls % 1000 == 0:
            self.training_stage += 1

    def get_data(self):
        if not self.args.azure:
            # Local data path
            data_dir = os.path.join("src/data", self.args.dataset.replace("-", "_"))
        else:
            data_dir = os.path.join("data", self.args.dataset.replace("-", "_"))
        train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        return train_data, val_data

    def get_batch(self, split, mode="train"):
        if mode == "train":
            self.update_training_stage()

        data = self.train_data if split == "train" else self.val_data

        if self.tokenizer.split_token_id is not None:
            return self._get_split_batch(data)
        else:
            return self._get_block_batch(data)

    def _get_split_batch(self, data):
        # Find all indices of the split token
        split_indices = np.where(data == self.tokenizer.split_token_id)[0]

        # Ensure we have enough complete exercises
        if len(split_indices) < self.args.batch_size + 1:
            raise ValueError(f"Not enough complete exercises in the dataset")

        # torch array with self.training_example_id times the batch size
        start_idx = torch.full((self.args.batch_size,), self.training_example_id)
        # start_idx = torch.randint(len(split_indices) - 1, (self.args.batch_size,))

        x = []
        y = []
        for idx in start_idx:
            start = split_indices[idx] + 1  # Start after the previous split token
            end = split_indices[idx + 1] + 1  # Include the current split token

            exercise = data[start:end]
            equal_sign_pos = np.where(exercise == self.tokenizer.vocab["="])[0][0]
            newline_pos = np.where(exercise == self.tokenizer.vocab["\n"])[0][0]

            x_seq = torch.from_numpy(exercise[: newline_pos + 1].astype(np.int64))
            y_seq = torch.full_like(
                x_seq, self.tokenizer.vocab["_"]
            )  # Fill with mask token

            # Only keep the tokens after '=' in y
            y_seq[equal_sign_pos + 1 :] = x_seq[equal_sign_pos + 1 :]

            x.append(x_seq[:-1])
            y.append(y_seq[1:])

        # Pad sequences to the length of the longest sequence in the batch
        max_len = max(len(seq) for seq in x)
        x = [
            F.pad(seq, (0, max_len - len(seq)), value=self.tokenizer.vocab["<pad>"])
            for seq in x
        ]
        y = [
            F.pad(seq, (0, max_len - len(seq)), value=self.tokenizer.vocab["<pad>"])
            for seq in y
        ]

        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device)

        return x, y

    def _get_block_batch(self, data):
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.args.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.args.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        x, y = x.to(self.device), y.to(self.device)
        return x, y


def main():
    args = parse_args()
    print("Arguments:")
    print(args)

    vocab, tokenizer = load_vocab(args.tokenizer)

    total_tokens = 0
    unique_tokens = set()
    epoch = 0

    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    mlflow.autolog()

    # Save all the config values to mlflow
    for k, v in vars(args).items():
        mlflow.log_param(k, v)

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        device = args.device

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
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

    data_loader = DataLoader(args, tokenizer, device)

    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        vocab_size=len(vocab),  # default to GPT-2 vocab size
        bias=args.bias,
    )

    best_val_loss = 1e9

    if args.init_from == "scratch":
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif args.init_from == "resume":
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k, v in model_args.items():
            assert (
                checkpoint_model_args[k] == v
            ), "Checkpoint model args do not match current args"
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        total_tokens = checkpoint["total_tokens"]
        unique_tokens = checkpoint["unique_tokens"]
    elif args.init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        model = GPT.from_pretrained(args.init_from, model_args)

    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2)
    )
    if args.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

    # compile the model
    if args.compile:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    X, Y = data_loader.get_batch("train", mode="eval")  # fetch the very first batch
    iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_tokens += args.batch_size * args.block_size
        unique_tokens.update(X.unique().cpu().numpy())

        if total_tokens >= data_loader.tokens_per_epoch * (epoch + 1):
            epoch += 1
            mlflow.log_metric("epochs_completed", epoch, step=iter_num)
            print(f"Completed epoch {epoch} at iteration {iter_num}")

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, data_loader, args, ctx, tokenizer, iter_num)
            print(f"Training stage: {data_loader.training_stage}")
            print(f"Learning rate: {lr}")
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            mlflow.log_metric("total_tokens", total_tokens, step=iter_num)
            mlflow.log_metric("unique_tokens", len(unique_tokens), step=iter_num)
            mlflow.log_metric("train_loss", losses["train"], step=iter_num)
            mlflow.log_metric("val_loss", losses["val"], step=iter_num)
            mlflow.log_metric("learning_rate", lr, step=iter_num)
            mlflow.log_metric(
                "current_epoch",
                epoch
                + (total_tokens % data_loader.tokens_per_epoch)
                / data_loader.tokens_per_epoch,
                step=iter_num,
            )
            if losses["val"] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": vars(args),
                        "total_tokens": total_tokens,
                        "unique_tokens": unique_tokens,
                    }
                    print(f"saving checkpoint to {args.out_dir}")
                    torch.save(checkpoint, os.path.join(args.out_dir, "ckpt.pt"))
        if iter_num == 0 and args.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / args.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = data_loader.get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        iter_num += 1

        # termination conditions
        if iter_num > args.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    with mlflow.start_run() as run:
        main()
