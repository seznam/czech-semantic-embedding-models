import argparse
from collections import defaultdict

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    Trainer,
    TrainingArguments,
)
from utils import (
    load_facebook,
    load_custom_embeddings_with_cache,
    EmbeddingForSequenceClassification,
)

K_FOLD = 10
NUM_EPOCH_FREEZE = 1
NUM_EPOCH_UNFREEZE = 3
BATCH_SIZE_PER_GPU = 32
LEARNING_RATE_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
LEARNING_RATE_HEAD_ONLY_GRID = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
SEED = 42
DEV_SIZE = 0.1


class FacebookDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoded_dataset = self._encode(data["post"].to_list())
        self.labels = torch.tensor(data["label"].to_list())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encoded_dataset.items()}
        return BatchEncoding({**item, "labels": self.labels[idx]})

    def _encode(self, texts):
        return self.tokenizer(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


class FacebookEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data, embedding_dict):
        self.emb_dict = embedding_dict
        data = self._filter_data(data)
        self.labels = torch.tensor(data["label"].to_list())
        self.embeddings = self._encode(data["post"].to_list())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}

    def _filter_data(self, data):
        df = data[data["post"].apply(lambda x: x in self.emb_dict)]
        if len(data) - len(df) > 0:
            print(
                f"WARNING: Embedding not found for {len(data) - len(df)}/{len(data)} samples. Ignoring."
            )
        return df

    def _encode(self, texts):
        return [self.emb_dict[text] for text in texts]


class FacebookMetric:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(
            predictions=predictions, references=labels, average="macro"
        )


def freeze_model_layers(model, freeze=True):
    for name, param in model.named_parameters():
        if not name.startswith("classifier") and "pooler" not in name:
            param.requires_grad = not freeze


def train_facebook(model, dataset_train, dataset_test, metric, n_epoch, learning_rate):
    training_args = TrainingArguments(
        seed=SEED,
        report_to="none",
        output_dir="data/facebook_experiments",
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        per_device_eval_batch_size=64,
        save_steps=99999,  # do not save
        do_train=True,
        num_train_epochs=n_epoch,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=metric,
        optimizers=[torch.optim.AdamW(model.parameters(), learning_rate), None],
    )
    trainer.train()
    return trainer


def evaluate_facebook(args):
    dataset = load_facebook()
    if args.debug:
        dataset = dataset.iloc[:200]

    if not args.eval_embeddings:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    facebook_metric = FacebookMetric(evaluate.load("f1"))

    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
    results = defaultdict(dict)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):

        if args.eval_embeddings:
            embeddings_dict = load_custom_embeddings_with_cache(
                args.eval_embeddings, args.debug
            )
            dataset_train = FacebookEmbeddingDataset(
                dataset.iloc[train_index], embeddings_dict
            )
            test = FacebookEmbeddingDataset(dataset.iloc[test_index], embeddings_dict)
            hidden_size = iter(embeddings_dict.values()).__next__().size(-1)
            model = EmbeddingForSequenceClassification(
                hidden_size=hidden_size, num_labels=3
            )
        else:
            dataset_train = FacebookDataset(dataset.iloc[train_index], tokenizer)
            test = FacebookDataset(dataset.iloc[test_index], tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_path, num_labels=3
            )

        train, dev = train_test_split(
            dataset_train, test_size=DEV_SIZE, random_state=SEED, shuffle=True
        )

        # FREEZED
        if args.num_epoch_freeze > 0:
            freeze_model_layers(model, freeze=True)
            trainer = train_facebook(
                model=model,
                dataset_train=train,
                dataset_test=dev,
                metric=facebook_metric,
                n_epoch=args.num_epoch_freeze,
                learning_rate=args.learning_rate if args.head_only else 1e-3,
            )
        # UNFREEZE
        if args.num_epoch_unfreeze > 0:
            freeze_model_layers(model, freeze=False)
            trainer = train_facebook(
                model=model,
                dataset_train=train,
                dataset_test=dev,
                metric=facebook_metric,
                n_epoch=args.num_epoch_unfreeze,
                learning_rate=args.learning_rate,
            )

        results[f"fold_{i}"] = {
            "test": trainer.evaluate(test)["eval_f1"],
            "dev": trainer.evaluate(dev)["eval_f1"],
        }
        print(results[f"fold_{i}"])

    df = pd.DataFrame(results, index=["test", "dev"])
    df["average"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)

    results = {
        "data": df,
        "score": df.loc["test", "average"],
        "std": df.loc["test", "std"],
        "dev_score": df.loc["dev", "average"],
    }
    return results


def main(args):
    torch.manual_seed(SEED)

    args.num_epoch_freeze = NUM_EPOCH_FREEZE
    args.num_epoch_unfreeze = NUM_EPOCH_UNFREEZE

    if args.model_path is None and not args.eval_embeddings:
        raise ValueError(
            "Either the model path or the eval_embeddings flag must be specified."
        )

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    if args.eval_embeddings:
        args.head_only = True

    if args.head_only:
        args.num_epoch_freeze += args.num_epoch_unfreeze
        args.num_epoch_unfreeze = 0

    results = defaultdict(dict)
    learning_rate_search_grid = (
        LEARNING_RATE_HEAD_ONLY_GRID if args.head_only else LEARNING_RATE_GRID
    )
    best_score, best_lr = -1, -1
    for lr in learning_rate_search_grid:
        args.learning_rate = lr
        results[lr] = evaluate_facebook(args)
        dev_score = results[lr]["dev_score"]
        if dev_score > best_score:
            best_score = dev_score
            best_lr = lr

    results["final_result"] = {
        "score": results[best_lr]["score"],
        "std": results[best_lr]["std"],
    }

    print_results(results)
    return results


def print_results(results, file=None):
    print("*" * 50, "CFD", "*" * 50, file=file)
    for lr in results:
        if lr == "final_result":
            continue
        print(f"Learning rate: {lr}", file=file)
        print(results[lr]["data"].round(4) * 100, file=file)
        print("-" * 50, file=file)

    print(
        f"FINAL SCORE: {results['final_result']['score'] * 100:.2f} +-{results['final_result']['std'] * 100:.2f}",
        file=file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("--head_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_embeddings", type=str)

    args = parser.parse_args()

    main(args)
