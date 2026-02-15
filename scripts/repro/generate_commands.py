#!/usr/bin/env python3
import argparse

def build_command(args):
    cfg_map = {
        "cifar10": "deep_al/configs/cifar10/al/RESNET18.yaml",
        "cifar100": "deep_al/configs/cifar100/al/RESNET18.yaml",
    }
    cfg = cfg_map[args.dataset]

    exp_name = args.exp_name
    if not exp_name:
        exp_name = f"repro_probcover_{args.dataset}_b{args.budget}_seed{args.seed}"

    cmd = [
        "python", "deep_al/tools/train_al.py",
        "--cfg", cfg,
        "--al", "bayes_misp",
        "--diff_method", "prob_cover",
        "--kernel_type", args.kernel_type,
        "--initial_delta", str(args.delta),
        "--exp-name", exp_name,
        "--initial_size", str(args.initial_size),
        "--budget", str(args.budget),
        "--seed", str(args.seed),
        "--eval_model_type", args.eval_model_type,
    ]
    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible ProbCover commands.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kernel_type", default="tophat")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--initial_size", type=int, default=0)
    parser.add_argument("--eval_model_type", default="from_images")
    parser.add_argument("--exp_name", default=None)
    args = parser.parse_args()

    if args.delta is None:
        args.delta = 0.75 if args.dataset == "cifar10" else 0.65

    print(build_command(args))


if __name__ == "__main__":
    main()
