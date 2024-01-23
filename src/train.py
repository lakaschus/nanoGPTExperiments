"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, SleepGPT

from types import SimpleNamespace

def main(args):

    # -----------------------------------------------------------------------------
    # Convert dict to namespace
    conf = SimpleNamespace(**args)
    # -----------------------------------------------------------------------------

    import mlflow
    from mlflow.models.signature import infer_signature
    
    mlflow.set_experiment(conf.experiment_name)
    mlflow.autolog(log_models=True)
    mlflow.log_artifacts('./outputs')
    
    # Get current date and time as string for run name
    from datetime import datetime
    now = datetime.now()
    # yy-mm-dd-hh-mm
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    
    # Set run name
    mlflow.set_tag("mlflow.runName", conf.experiment_name + "_" + dt_string)

    # Save all the config values to mlflow
    for k, v in args.items():
        mlflow.log_param(k, v)

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=conf.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
    if master_process:
        print("Create out_dir...")
        # Make sure the out_dir is located in the same directory as this script
        out_dir = os.path.join(os.path.dirname(__file__), conf.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        # Verify that the out_dir was created
        assert os.path.exists(out_dir)
    torch.manual_seed(conf.base_seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in conf.device else 'cpu' # for later use in torch.autocast
    # note: float16 would require us to change the code to use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type,dtype=ptdtype)
    # poor man's data loader, TODO evaluate need for actual DataLoader
    data_dir = os.path.join('data', conf.dataset)
    # If .bin files are not present, run the data preprocessing script first.
    if not os.path.exists(os.path.join(data_dir, 'train.bin')):
        import subprocess
        print("Prepare dataset...")
        subprocess.run(['python', f'data/{conf.dataset}/prepare.py', conf.dataset])
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - conf.block_size, (conf.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+conf.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+conf.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(conf.device), y.to(conf.device)
        return x, y
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"vocab_size = {vocab_size} (from {meta_path})")
    else:
        print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
        vocab_size = 50257
    # model init
    model_args = dict(n_layer=conf.n_layer, n_head=conf.n_head, n_embd=conf.n_embd, block_size=conf.block_size,
                    dropout=conf.dropout, vocab_size=vocab_size, bias=conf.bias, weight_init_std=conf.weight_init_std)
    if conf.init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        if conf.model == 'sleepgpt':
            model = SleepGPT(gptconf)
        else:
            model = GPT(gptconf)
    elif conf.init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k, v in model_args.items():
            assert checkpoint_model_args[k] == v, "for now"
            # TODO: think through how passed in params should interact with checkpoint params
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif conf.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {conf.init_from}")
        assert conf.bias, "GPT-2 models have bias, so we can't use bias=False"
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=conf.dropout)
        model = GPT.from_pretrained(conf.init_from, override_args)
        # read off and override the GPT sizing model args from the model config
        model_args['n_layer'] = model.config.n_layer
        model_args['n_head'] = model.config.n_head
        model_args['n_embd'] = model.config.n_embd
    # crop down the model block size if desired
    if conf.block_size < model.config.block_size:
        model.crop_block_size(conf.block_size)
    model.to(conf.device)
    # optimizer
    optimizer = model.configure_optimizers(conf.weight_decay, conf.learning_rate, (conf.beta1, conf.beta2))
    if conf.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    # compile the model
    if conf.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(conf.eval_iters)
            for k in range(conf.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter):
        # 1) linear warmup for warmup_iters steps
        if iter < conf.warmup_iters:
            return conf.learning_rate * iter / conf.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > conf.lr_decay_iters:
            return conf.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - conf.warmup_iters) / (conf.lr_decay_iters - conf.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return conf.min_lr + coeff * (conf.learning_rate - conf.min_lr)
    # training loop
    t0 = time.time()
    while True:
        # determine the learning rate for this iteration
        if conf.lr_decay_iters:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = conf.learning_rate
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % conf.eval_interval == 0 and master_process:
            losses = estimate_loss()
            # Use mlflow to log metrics to 4 digits after the decimal point
            mlflow.log_metric("val_loss", losses['val'], step=iter_num)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or conf.always_save_checkpoint:
                best_val_loss = losses['val']
                raw_model = model.module if ddp else model
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': conf,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and conf.eval_only:
            break
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(conf.gradient_accumulation_steps):
            X, Y = get_batch('train')
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == conf.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
            loss.backward()
        if conf.grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % conf.log_interval == 0 and master_process:
            
            lossf = loss.item() # loss as float. TODO note CPU-GPU sync! profile, make sure not too slow
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            mlflow.log_metric("train_loss", losses['train'], step=iter_num)
        iter_num += 1
        # termination conditions
        if iter_num > conf.max_iters:
            break
    if ddp:
        destroy_process_group()
        
    # create the signature by inferring it from the datasets
    # signature = infer_signature(train_data)
    # mlflow.pytorch.log_model(raw_model, "./outputs", signature=signature)

if __name__ == '__main__':
    pass