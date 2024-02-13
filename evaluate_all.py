import argparse
import os
import pprint
from datetime import datetime
from importlib import import_module
from pathlib import Path

TASKS = ["sts", "costra", "facebook", "ctdc", "relevance"]


def main(args):
    results = {}
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for task in TASKS:
        task_script = import_module("evaluate_" + task)
        results[task] = getattr(task_script, "main")(args)

        output_dir = Path("results")
        output_dir.mkdir(parents=False, exist_ok=True)
        output_path = output_dir / (eval_timestamp + args.name + ".txt")
        with open(output_path, "at") as log_file:
            getattr(task_script, "print_results")(results[task], log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("--head_only", action="store_true")
    parser.add_argument("-n", "--name", type=str, default="")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_data", type=int, required=False)
    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument(
        "-p",
        "--pooling",
        type=str,
        default="cls",
        help="one in ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']",
    )
    parser.add_argument("--eval_embeddings", type=str)
    args = parser.parse_args()
    if args.name == "":
        if args.eval_embeddings:
            args.name = args.eval_embeddings.split("/")[-1]
        else:
            args.name = args.model_path.split("/")[-1]
    args.name = "_" + args.name

    main(args)
