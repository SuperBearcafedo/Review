from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from experiments.exp_long_term_forecasting import Experiment

SEQ_LEN = 96
PRED_LEN = 12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KUMA on PEMS03")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--random_seed", type=int, default=2025)

    parser.add_argument("--root_path", default="./dataset/PEMS")
    parser.add_argument("--data_path", default="PEMS03.npz")
    parser.add_argument("--enc_in", type=int, default=358)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", choices=("relu", "gelu"), default="gelu")
    parser.add_argument("--disable_norm", action="store_true")

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_setting(args: argparse.Namespace) -> str:
    model_id = args.model_id or f"PEMS03_{args.seq_len}_{args.pred_len}"
    return (
        f"{model_id}_KUMA_sl{args.seq_len}_pl{args.pred_len}"
        f"_dm{args.d_model}_el{args.e_layers}_seed{args.random_seed}"
    )


def main() -> None:
    args = build_parser().parse_args()
    args.seq_len = SEQ_LEN
    args.pred_len = PRED_LEN
    if args.d_model < 4 or args.d_model % 4 != 0:
        raise ValueError("d_model must be a positive multiple of 4 for the UMA hierarchy.")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max_train_batches must be positive.")
    if args.max_eval_batches is not None and args.max_eval_batches < 1:
        raise ValueError("max_eval_batches must be positive.")

    args.use_norm = not args.disable_norm
    args.use_gpu = torch.cuda.is_available() and not args.cpu
    set_seed(args.random_seed)

    setting = experiment_setting(args)
    print(f"Device request: {'cuda:' + str(args.gpu) if args.use_gpu else 'cpu'}")
    print(f"Experiment: {setting}")

    experiment = Experiment(args)
    if args.mode == "train":
        experiment.train(setting)
        experiment.test(setting, load_checkpoint=False)
    else:
        experiment.test(setting, load_checkpoint=True)


if __name__ == "__main__":
    main()
