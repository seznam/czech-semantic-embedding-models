import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torchmetrics import F1Score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    Trainer,
    TrainingArguments,
)
from utils import (
    load_ctdc,
    EmbeddingForSequenceClassification,
    load_custom_embeddings_with_cache,
)

K_FOLD = 5
DEV_SIZE = 0.1
NUM_EPOCH_FREEZE = 1
NUM_EPOCH_UNFREEZE = 5
BATCH_SIZE = 32
LEARNING_RATE_GRID = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
LEARNING_RATE_GRID_HEAD_ONLY = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
SEED = 42


class CTDCDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.encoded_dataset = self._encode(data["text"].to_list())
        self.data = data.drop("text", axis=1).astype(float)
        self.labels = torch.tensor(self.data.to_numpy())

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


class CTDCEmbeddingDataset(torch.utils.data.Dataset):

    # if the first number of {COMPARE_LIMIT} characters in two string match, they are considered equal
    COMPARE_LIMIT = 200

    def __init__(self, data, embeddings_dict):
        self.emb_dict = embeddings_dict
        self.emb_dict = self._fix_emb_dict()
        data = self._filter_data(data)

        self.embeddings = self._encode(data["text"].to_list())
        self.labels = torch.tensor(data.drop("text", axis=1).astype(float).to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return BatchEncoding(
            {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}
        )

    def _filter_data(self, data):
        df = data[
            data["text"].apply(lambda x: x[: self.COMPARE_LIMIT] in self.emb_dict)
        ]
        if len(data) - len(df) > 0:
            print(
                f"WARNING: Embedding not found for {len(data) - len(df)}/{len(data)} samples. Ignoring."
            )
        return df

    def _encode(self, texts):
        return [self.emb_dict[text[: self.COMPARE_LIMIT]] for text in texts]

    def _fix_emb_dict(self):
        new_dict = {}
        for key in self.emb_dict.keys():
            new_dict[key[: self.COMPARE_LIMIT]] = self.emb_dict[key]
        return new_dict


def freeze_everything_except_head(model, freeze=True):
    for name, param in model.named_parameters():
        if not name.startswith("classifier") and "pooler" not in name:
            param.requires_grad = not freeze


def make_ctdc_trainer(model, dataset_train, dataset_test, lr, n_epochs):
    training_args = TrainingArguments(
        seed=SEED,
        report_to="none",
        output_dir="data/ctdc_experiments",
        num_train_epochs=n_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        do_train=True,
        save_steps=99999,  # do not save
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        optimizers=[torch.optim.AdamW(model.parameters(), lr), None],
    )
    return trainer


def do_ctdc_fold(tokenizer, dataset, train_index, test_index, args):
    if args.eval_embeddings:
        embeddings_dict = load_custom_embeddings_with_cache(
            args.eval_embeddings, args.debug
        )
        fold_train = CTDCEmbeddingDataset(dataset.iloc[train_index], embeddings_dict)
        test = CTDCEmbeddingDataset(dataset.iloc[test_index], embeddings_dict)
        hidden_size = iter(embeddings_dict.values()).__next__().size(-1)
        model = EmbeddingForSequenceClassification(
            hidden_size=hidden_size, num_labels=37
        )
    else:
        fold_train = CTDCDataset(dataset.iloc[train_index], tokenizer)
        test = CTDCDataset(dataset.iloc[test_index], tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=37
        )
    train, dev = train_test_split(
        fold_train, test_size=DEV_SIZE, random_state=SEED, shuffle=True
    )

    if args.num_epoch_freeze > 0:
        freeze_everything_except_head(model, freeze=True)
        trainer = make_ctdc_trainer(
            model=model,
            dataset_train=train,
            dataset_test=dev,
            lr=args.lr,
            n_epochs=args.num_epoch_freeze,
        )
        trainer.train()
    if args.num_epoch_unfreeze > 0:
        freeze_everything_except_head(model, freeze=False)
        trainer = make_ctdc_trainer(
            model=model,
            dataset_train=train,
            dataset_test=dev,
            lr=args.lr,
            n_epochs=args.num_epoch_unfreeze,
        )
        trainer.train()

    test_preds = trainer.predict(test)
    dev_preds = trainer.predict(dev)

    return dev_preds, test_preds


def find_best_threshold(eval_preds_list):
    max_score, best_th, best_scores = -1, 0, None
    for threshold in np.linspace(0, 1, 50):
        scores = eval_threshold(eval_preds_list, threshold)
        score = sum(scores) / len(scores)
        if score > max_score:
            max_score = score
            best_th = threshold
            best_scores = scores

    return best_scores, best_th


def eval_threshold(eval_preds_list, th):
    f1 = F1Score(task="multilabel", threshold=th, num_labels=37, average="micro")
    scores = []
    for eval_preds in eval_preds_list:
        logits, labels = eval_preds
        scores.append(f1(torch.from_numpy(logits), torch.from_numpy(labels)).item())

    return scores


def eval_ctdc(args):
    dataset = load_ctdc()
    if args.debug:
        dataset = dataset.iloc[:200]
    tokenizer = (
        AutoTokenizer.from_pretrained(args.tokenizer_path)
        if not args.eval_embeddings
        else None
    )
    results = defaultdict(lambda: defaultdict(dict))  # double default dictionary
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
    learning_rate_grid = (
        LEARNING_RATE_GRID_HEAD_ONLY if args.head_only else LEARNING_RATE_GRID
    )
    for n_fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        for lr in learning_rate_grid:
            args.lr = lr
            # train model
            dev_predictions, test_predictions = do_ctdc_fold(
                tokenizer=tokenizer,
                dataset=dataset,
                train_index=train_index,
                test_index=test_index,
                args=args,
            )
            # save dev preds
            results[lr][n_fold]["dev_predictions"] = (
                dev_predictions.predictions,
                dev_predictions.label_ids,
            )
            # save test preds
            results[lr][n_fold]["test_predictions"] = (
                test_predictions.predictions,
                test_predictions.label_ids,
            )

    # find the best hyper parameters
    best_score, best_lr, best_th = -1, -1, -1
    for lr in learning_rate_grid:
        # find the best threshold for specific LR across all folds
        dev_lr_results = [
            results[lr][n_fold]["dev_predictions"] for n_fold in results[lr]
        ]
        dev_scores, dev_th = find_best_threshold(dev_lr_results)

        # eval test with the best threshold (found on dev)
        test_lr_results = [
            results[lr][n_fold]["test_predictions"] for n_fold in results[lr]
        ]
        test_scores = eval_threshold(test_lr_results, dev_th)

        df = pd.DataFrame(
            {"dev": dev_scores, "test": test_scores},
            index=[f"fold_{x}" for x in range(len(dev_scores))],
        ).T
        df["average"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)

        results[lr] = {
            "data": df,
            "threshold": dev_th,
        }

        dev_score = sum(dev_scores) / len(dev_scores)
        if dev_score > best_score:
            best_score, best_lr, best_th = dev_score, lr, dev_th

    results["final_result"] = {
        "score": results[best_lr]["data"].loc["test", "average"],
        "learning_rate": best_lr,
        "threshold": best_th,
        "std": results[best_lr]["data"].loc["test", "std"],
    }

    print_results(results)
    return results


def print_results(results, file=None):
    print("*" * 50, "CTDC", "*" * 50, file=file)
    for lr in results:
        if lr == "final_result":
            continue
        print(f"Learning rate {lr}:", file=file)
        print(results[lr]["data"].round(4) * 100, file=file)
        print(f"\tThreshold: {results[lr]['threshold']}", file=file)
        print("-" * 50, file=file)
    print(results["final_result"], file=file)
    print(file=file)
    print(
        f'FINAL SCORE: {results["final_result"]["score"] * 100:.2f}',
        f'+-{results["final_result"]["std"] * 100:.2f}',
        file=file,
    )


def main(args):
    torch.manual_seed(SEED)
    args.num_epoch_freeze = NUM_EPOCH_FREEZE
    args.num_epoch_unfreeze = NUM_EPOCH_UNFREEZE

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    if args.model_path is None and not args.eval_embeddings:
        raise ValueError(
            "Either the model path or the eval_embeddings flag must be specified."
        )

    if args.eval_embeddings:
        args.head_only = True

    if args.head_only:
        args.num_epoch_freeze += args.num_epoch_unfreeze
        args.num_epoch_unfreeze = 0

    results = eval_ctdc(args)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("--head_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_embeddings", type=str)
    args = parser.parse_args()

    main(args)
