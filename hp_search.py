import subprocess as sp
import argparse
import os
import torch
import traceback
import sys, time
import numpy as np
from train import run
from train import get_args as get_train_args

def _classify_runtime_error(e: RuntimeError) -> str:
    s = str(e).lower()
    if ("out of memory" in s) or ("cuda oom" in s) or ("cublas_status_alloc_failed" in s) or ("cudnn_status_alloc_failed" in s):
        return "oom"
    if ("illegal memory access" in s) or ("device-side assert" in s) or ("an illegal memory access" in s):
        return "cuda_illegal"
    if "cublas" in s:
        return "cublas"
    if "cudnn" in s:
        return "cudnn"
    if "nccl" in s or "peer access" in s:
        return "nccl"
    if "size mismatch" in s or "shape" in s or "mat1 and mat2 shapes" in s:
        return "shape"
    return "other"

def _dump_cuda_state():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[CUDA] cuda.is_available() = False", flush=True)
            return
        dc = torch.cuda.device_count()
        print(f"[CUDA] device_count={dc}, visible='{os.getenv('CUDA_VISIBLE_DEVICES')}'", flush=True)
        for i in range(dc):
            try:
                name = torch.cuda.get_device_name(i)
                alloc = torch.cuda.memory_allocated(i) // (1024**2)
                reserv = torch.cuda.memory_reserved(i) // (1024**2)
                maxalloc = torch.cuda.max_memory_allocated(i) // (1024**2)
                print(f"[CUDA:{i}] {name} | alloc={alloc}MB reserved={reserv}MB max_alloc={maxalloc}MB", flush=True)
            except Exception as ie:
                print(f"[CUDA:{i}] <probe failed> {ie}", flush=True)
    except Exception as ie:
        print(f"[CUDA] state dump failed: {ie}", flush=True)

def dump_args(ns, tag="ARGS"):
    try:
        import os
        kv = {k: getattr(ns, k) for k in vars(ns)}
        print(f"[{tag}] " + " ".join(f"{k}={repr(v)}" for k, v in sorted(kv.items())), flush=True)
        if hasattr(ns, "input_file"):
            print(f"[{tag}] input_file={ns.input_file} exists? {os.path.exists(ns.input_file)}", flush=True)
    except Exception as e:
        print(f"[{tag}] <failed to dump args: {e}>", flush=True)

# 前面进来一次参数了，为什么这里还要有一遍
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default='examples/gym_data/tensat/cyclic/resnet50.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--acyclic', action='store_true', default=False)
    return parser.parse_args()


def call_command(args):
    print("[call_command] enter", flush=True)
    dump_args(args, tag="call_command")
    try:
        log = run(args)

    # BSC main error occurs
    except RuntimeError as e:
        # 1) 打印异常信息（类型 + 完整堆栈）
        etype = type(e).__name__
        msg = str(e)
        kind = _classify_runtime_error(e)
        print("\n" + "="*80, flush=True)
        print(f"[RuntimeError] kind={kind} type={etype}", flush=True)
        print(f"[RuntimeError] message: {msg}", flush=True)
        print(f"[RuntimeError] args: {getattr(e, 'args', None)}", flush=True)
        print("-"*80, flush=True)
        # 完整堆栈（包含 causes / context）
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)), flush=True)
        print("-"*80, flush=True)

        # 2) 打印运行时上下文（GPU/环境/关键参数）
        print(f"[CTX] gpus={getattr(args, 'gpus', None)} "
              f"batch_size={getattr(args, 'batch_size', None)} "
              f"gpu_ids='{os.getenv('CUDA_VISIBLE_DEVICES')}'", flush=True)
        _dump_cuda_state()
        print("="*80 + "\n", flush=True)

        # 3) 针对 OOM 的回退；非 OOM 不要盲目减 batch
        if kind == "oom":
            if args.batch_size is None:
                args.batch_size = 64
            elif args.batch_size > 1:
                # 对半减，但至少为 1
                args.batch_size = max(1, args.batch_size // 2)
            else:
                return None

            # 若多卡，保证 batch_size 能被 gpus 整除（向下对齐；至少每卡1个）
            if getattr(args, "gpus", 0) and args.gpus > 1:
                snapped = (args.batch_size // args.gpus) * args.gpus
                if snapped < args.gpus:
                    # 实在太小了：改为每卡1个
                    snapped = args.gpus
                if snapped != args.batch_size:
                    print(f"[BATCH] snap {args.batch_size} -> {snapped} for gpus={args.gpus}", flush=True)
                    args.batch_size = snapped

            # 递归重试（注意：不再把 args.batch_size 复位为 None）
            return call_command(args)

        # 4) 非 OOM：直接返回 None（或改为上抛），避免“误降 batch 导致归零”
        return None
    except ValueError as ve:
        # print(f"Caught ValueError: {str(ve)}")
        return None
    except Exception as ex:
        # debug增加报错信息
        print(f"Caught an unexpected exception: {repr(ex)}", flush=True)
        print("".join(traceback.format_exception(type(ex), ex, ex.__traceback__)), flush=True)
        return None
    return log


if __name__ == "__main__":
    args = get_args()

    best_hp = None
    best_loss = float('inf')
    best_time = float('inf')

    optimizers = ['rmsprop', 'adamw']
    lrs = [1e-1, 1e-2]
    assumptions = ['independent', 'correlated', 'hybrid']
    regs = [1e-2, 1e-4]
    for optimizer in optimizers:
        for lr in lrs:
            for assumption in assumptions:
                for reg in regs:
                    train_args = get_train_args(default=True)
                    train_args.num_steps = 100
                    train_args.input_file = args.input_file
                    train_args.gpus = args.gpus
                    train_args.batch_size = args.batch_size

                    train_args.optimizer = optimizer
                    train_args.assumption = assumption
                    train_args.base_lr = lr
                    train_args.regularizer = reg
                    train_args.acyclic = args.acyclic

                    print(
                        f'optimizer: {optimizer}, lr: {lr}, assumption: {assumption}, reg: {reg}'
                    )
                    log = call_command(train_args)
                    if log is None:
                        continue
                    min_loss = min(log['inference_loss'])
                    min_iter = np.argmin(log['inference_loss'])
                    time = log['time'][min_iter]
                    print(f'Min loss: {min_loss}, time: {time}')

                    if (min_loss < best_loss) or (min_loss == best_loss
                                                  and time < best_time):
                        best_hp = {
                            'optimizer': optimizer,
                            'lr': lr,
                            'assumption': assumption,
                            'reg': reg
                        }
                        best_loss = min_loss
                        best_time = time
    print(f'Best hyperparameters: {best_hp}, loss: {best_loss}')
