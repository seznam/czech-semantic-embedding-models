import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Literal, Optional

import torch
from sentence_transformers import (
    InputExample,
    LoggingHandler,
    SentenceTransformer,
    losses,
)
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.models import Pooling, Transformer
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torchmetrics.retrieval import RetrievalMetric
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
from utils import similarity_function, load_dareczech, load_custom_embeddings_with_cache

BATCH_SIZE = 25
LEARNING_RATE_SEARCH_GRID = [1e-6, 5e-6, 2e-5, 5e-5]

logger = logging.getLogger(__name__)


def _binarize_target(target: Tensor) -> Tensor:
    return torch.where(target > 0.5, 1, 0)


def _identity_fn(target: Tensor) -> Tensor:
    return target


def precision_at_k(preds: Tensor, target: Tensor, k: int = 10) -> Tensor:
    preds, target = _check_retrieval_functional_inputs(preds, target)

    k = min(k, preds.shape[-1])

    if not (isinstance(k, int) and k > 0):
        raise ValueError("`k` has to be a positive integer or None")

    if not target.sum():
        return torch.tensor(0.0, device=preds.device)

    relevant = target[preds.topk(k, dim=-1)[1]].sum().float()
    return relevant / k


class PrecisionAtK(RetrievalMetric):
    def __init__(
        self,
        k: int = 10,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        target_manipulation: Optional[Callable] = _binarize_target,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action, ignore_index=ignore_index, **kwargs
        )
        if not (isinstance(k, int) and k > 0):
            raise ValueError("`k` has to be a positive integer.")
        self.k = k
        self.target_manipulation = target_manipulation or _identity_fn

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:  # type: ignore
        """Check shape, check and convert dtypes, flatten and add to accumulators."""
        target = self.target_manipulation(target)
        super().update(preds, target, indexes)

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return precision_at_k(preds, target, self.k)

    def __str__(self):
        return f"precision_at_{self.k}"


class RelevanceEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data, embedding_dict):
        self.emb_dict = embedding_dict
        data = self._filter_data(data)
        self.embeddings = self._encode(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx]}

    def _filter_data(self, data):
        df = [text for text in data if text in self.emb_dict]
        if len(data) - len(df) > 0:
            print(
                f"WARNING: Embedding not found for {len(data) - len(df)}/{len(data)} samples. Ignoring."
            )
        return df

    def _encode(self, texts):
        return [self.emb_dict[text] for text in texts]


class RelevanceEvaluator(SentenceEvaluator):
    def __init__(
        self,
        indexes,
        queries,
        documents,
        labels,
        precision_at_k: int = 10,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
    ):
        self.indexes = torch.tensor(indexes)
        self.queries = queries
        self.documents = documents
        self.labels = torch.tensor(labels)
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.precision_at_k = precision_at_k

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self.csv_headers.append("cos_sim-Precision@{}".format(precision_at_k))

    def __call__(
        self,
        model,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> float:
        if epoch != -1:
            out_txt = (
                " after epoch {}:".format(epoch)
                if steps == -1
                else " in epoch {} after {} steps:".format(epoch, steps)
            )
        else:
            out_txt = ":"

        logger.info(
            "Information Retrieval Evaluation on " + self.name + " dataset" + out_txt
        )

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            output_data.append(scores["precision@k"][self.precision_at_k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        return max([scores["precision@k"][self.precision_at_k]])

    def compute_metrices(
        self,
        model,
        corpus_model=None,
        corpus_embeddings: Tensor = None,
        query_embeddings: Tensor = None,
    ) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        if query_embeddings is None:
            query_embeddings = model.encode(
                self.queries,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )
        if corpus_embeddings is None:
            corpus_embeddings = model.encode(
                self.documents,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )
        preds = cosine_similarity(query_embeddings, corpus_embeddings).cpu()
        metric = PrecisionAtK(k=10)
        metric.update(preds, self.labels, self.indexes)
        return {"precision@k": {10: metric.compute().item()}}

    def output_scores(self, scores):
        for k in scores["precision@k"]:
            logger.info(
                "precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100)
            )

    def bootstrap_resampling(self, model, n_resample=50):
        query_embeddings = model.encode(
            self.queries,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            convert_to_tensor=True,
        )
        corpus_embeddings = model.encode(
            self.documents,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            convert_to_tensor=True,
        )
        results = torch.empty(size=(n_resample,), device=model.device)
        for n in range(n_resample):
            sample = torch.randint(
                0, len(query_embeddings), size=(len(query_embeddings),)
            )
            preds = similarity_function(
                query_embeddings[sample], corpus_embeddings[sample], method="cs"
            )
            metric = PrecisionAtK(k=10)
            metric.update(
                preds,
                self.labels.to(model.device)[sample],
                self.indexes.to(model.device)[sample],
            )
            results[n] = metric.compute()
        return torch.mean(results).cpu().item(), torch.std(results).cpu().item()


def get_information_retrieval_evaluator(df, name=""):
    df["query_ids"] = df["query"].factorize()[0]
    return RelevanceEvaluator(
        indexes=df["query_ids"].to_list(),
        queries=df["query"].to_list(),
        documents=df["doc"].to_list(),
        labels=df["label"].to_list(),
        precision_at_k=10,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        name=name,
    )


def train_relevance(args):
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    n_data = getattr(args, "n_data", None)
    num_epoch = getattr(args, "num_epoch", 2)

    word_embedding_model = Transformer(
        args.model_path, max_seq_length=128, tokenizer_name_or_path=args.tokenizer_path
    )
    pooling = Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=args.pooling,
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling], device="cuda:0"
    )

    train_dataset = load_dareczech(
        "train", debug=args.debug, n_data=n_data, random_state=args.seed
    )
    train_samples = [
        InputExample(texts=[q, d], label=float(l))
        for q, d, l in zip(
            train_dataset["query"], train_dataset["doc"], train_dataset["label"]
        )
    ]

    train_dataloader = DataLoader(
        train_samples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if not args.debug else False,
        prefetch_factor=3 if not args.debug else None,
        num_workers=3 if not args.debug else 0,
    )

    train_loss = losses.CosineSimilarityLoss(model)

    dev_evaluator = get_information_retrieval_evaluator(
        load_dareczech("dev", debug=args.debug), "dev"
    )

    exp_name = str(int(datetime.now().timestamp()))
    output_path = "data/relevance_experiments/" + exp_name
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        scheduler="warmupcosine",
        optimizer_params={"lr": args.learning_rate, "eps": 1e-6},
        evaluation_steps=4000,
        output_path=output_path,
        save_best_model=True,
        epochs=num_epoch,
        warmup_steps=2000,
    )
    model.cpu()
    del model
    return output_path


def evaluate_relevance(
    output_path,
    dataset_type: Literal["test", "dev"] = "test",
    tokenizer_path=None,
    debug=False,
    n_resample=150,
    pooling_mode="cls",
):
    word_embedding_model = Transformer(
        output_path, max_seq_length=128, tokenizer_name_or_path=tokenizer_path
    )
    pooling = Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling], device="cuda:0"
    )
    model.eval()

    # bootstrap resampling
    dataset = load_dareczech(dataset_type, debug)
    test_evaluator = get_information_retrieval_evaluator(
        dataset, dataset_type + "_result"
    )
    average, std = test_evaluator.bootstrap_resampling(model, n_resample=n_resample)
    results = {
        "average": average,
        "std": std,
    }
    return results


def evaluate_relevance_embeddings(
    embedding_dict,
    dataset_type: Literal["test", "dev"] = "test",
    debug=False,
    n_resample=150,
):
    dataset = load_dareczech(dataset_type, debug)
    query_dataset = RelevanceEmbeddingDataset(dataset["query"], embedding_dict)
    document_dataset = RelevanceEmbeddingDataset(dataset["doc"], embedding_dict)

    query_embeddings = torch.stack([item["embeddings"] for item in query_dataset])
    corpus_embeddings = torch.stack([item["embeddings"] for item in document_dataset])
    labels = torch.tensor(dataset["label"].to_list())
    indexes = torch.from_numpy(dataset["query"].factorize()[0])

    results = torch.empty(size=(n_resample,))
    for n in range(n_resample):
        sample = torch.randint(0, len(query_embeddings), size=(len(query_embeddings),))
        preds = similarity_function(
            query_embeddings[sample], corpus_embeddings[sample], method="cs"
        )
        metric = PrecisionAtK(k=10)
        metric.update(preds, labels[sample], indexes[sample])
        results[n] = metric.compute()
    average, std = torch.mean(results).cpu().item(), torch.std(results).cpu().item()
    results = {
        "average": average,
        "std": std,
    }
    return results


def print_results(results, file=None):
    print("*" * 50, "RELEVANCE", "*" * 50, file=file)
    for lr in results:
        if lr == "final_result":
            continue
        print(f"Learning rate: {lr}", file=file)
        print(
            f"Score: {results[lr]['average'] * 100:.2f} +-{results[lr]['std'] * 100:.2f}",
            file=file,
        )
        print("-" * 50, file=file)
    print(
        f"FINAL SCORE: {results['final_result']['average'] * 100:.2f} +-{results['final_result']['std'] * 100:.2f}",
        file=file,
    )


def main(args):
    torch.manual_seed(args.seed)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    if args.eval_embeddings:
        args.head_only = True

    if args.head_only:

        if args.eval_embeddings:
            embedding_dict = load_custom_embeddings_with_cache(
                args.eval_embeddings, args.debug
            )
            results = {
                "final_result": evaluate_relevance_embeddings(
                    embedding_dict, "test", debug=args.debug
                )
            }
        else:
            results = {
                "final_result": evaluate_relevance(
                    args.model_path,
                    debug=args.debug,
                    tokenizer_path=args.tokenizer_path,
                    pooling_mode=args.pooling,
                )
            }

        print_results(results)
        return results

    results = defaultdict(dict)
    best_score, best_output_path = -1, None
    for lr in LEARNING_RATE_SEARCH_GRID:
        args.learning_rate = lr

        output_path = train_relevance(args)
        dev_result = evaluate_relevance(
            output_path,
            dataset_type="dev",
            debug=args.debug,
            tokenizer_path=args.tokenizer_path,
            n_resample=25,
        )
        results[lr] = dev_result
        dev_score = dev_result["average"]

        if dev_score > best_score:
            best_output_path = output_path
            best_score = dev_score

    results["final_result"] = evaluate_relevance(
        best_output_path,
        dataset_type="test",
        debug=args.debug,
        tokenizer_path=args.tokenizer_path,
        n_resample=100,
    )
    print_results(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("--head_only", action="store_true")
    parser.add_argument(
        "-p",
        "--pooling",
        type=str,
        default="cls",
        help="one in ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_data", type=int, required=False)
    parser.add_argument("--num_epoch", type=int, default=2)
    parser.add_argument("--eval_embeddings", type=str)
    args = parser.parse_args()

    main(args)
