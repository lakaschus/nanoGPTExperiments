import os
import time
import math
import argparse
from contextlib import nullcontext
from transformers import GPT2Tokenizer

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import tempfile
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

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
        default=10,
        help="Number of tokens to sample during evaluation",
    )

    # data
    parser.add_argument(
        "--dataset", type=str, default="openwebtext_10k", help="Dataset to use"
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
        "--learning_rate", type=float, default=6e-4, help="Learning rate"
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
        "--compile", action="store_true", help="Use PyTorch 2.0 to compile the model"
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
def estimate_loss(model, get_batch, args, ctx, tokenizer, iter_num):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Sample tokens
    x = torch.zeros((1, 1), dtype=torch.long).to(args.device)
    sample_tokens = []
    for _ in range(args.num_sample_tokens):
        with ctx:
            logits, _ = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        x_next = torch.multinomial(probs, num_samples=1)
        sample_tokens.append(x_next.item())
        x = torch.cat((x, x_next), dim=1)

    # Decode sampled tokens
    decoded_sample = tokenizer.decode(sample_tokens)
    print(f"Sampled text: {decoded_sample}\n")

    # Log sampled text to MLflow
    mlflow.log_text(decoded_sample, f"sampled_text_iter_{iter_num}.txt")

    model.train()
    return out


def load_vocab():
    """
    Load the GPT-2 tokenizer and return its vocabulary.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    return vocab, tokenizer


def main():
    args = parse_args()
    print("Arguments:")
    print(args)

    vocab, tokenizer = load_vocab()

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

    def get_data(args):
        if not args.azure:
            # Local data path
            data_dir = os.path.join("data", args.dataset)
            train_data = np.memmap(
                os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )
            val_data = np.memmap(
                os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
            )
        else:
            # Azure ML data path
            print("Using Azure ML dataset")
            try:
                ml_client = MLClient.from_config(credential=DefaultAzureCredential())
                dataset = ml_client.data.get(name=args.dataset, version="latest")

                base_path = "./data/"
                train_path = base_path + "train.bin"
                val_path = base_path + "val.bin"

                print(train_path)

                artifact_utils.download_artifact_from_aml_uri(
                    uri=dataset.path,
                    destination=base_path,
                    datastore_operation=ml_client.datastores,
                )

                # Read the data
                train_data = np.fromfile(train_path, dtype=np.uint16)
                val_data = np.fromfile(val_path, dtype=np.uint16)

                return train_data, val_data

            except Exception as e:
                print(f"Error accessing Azure ML dataset: {e}")
                raise

        # Read the data (for both local and Azure cases)
        train_data = np.fromfile(train_path, dtype=np.uint16)
        val_data = np.fromfile(val_path, dtype=np.uint16)

        return train_data, val_data

    train_data, val_data = get_data(args)

    tokens_per_epoch = len(train_data)

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + args.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + args.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        x, y = x.to(device), y.to(device)
        return x, y

    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        vocab_size=50257,  # default to GPT-2 vocab size
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

    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_tokens += args.batch_size * args.block_size
        unique_tokens.update(X.unique().cpu().numpy())

        if total_tokens >= tokens_per_epoch * (epoch + 1):
            epoch += 1
            mlflow.log_metric("epochs_completed", epoch, step=iter_num)
            print(f"Completed epoch {epoch} at iteration {iter_num}")

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, get_batch, args, ctx, tokenizer, iter_num)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            mlflow.log_metric("total_tokens", total_tokens, step=iter_num)
            mlflow.log_metric("unique_tokens", len(unique_tokens), step=iter_num)
            mlflow.log_metric("train_loss", losses["train"], step=iter_num)
            mlflow.log_metric("val_loss", losses["val"], step=iter_num)
            mlflow.log_metric(
                "current_epoch",
                epoch + (total_tokens % tokens_per_epoch) / tokens_per_epoch,
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
            X, Y = get_batch("train")
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
