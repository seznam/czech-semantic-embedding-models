import argparse
import numpy as np
from collections import defaultdict

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from tqdm import tqdm
from utils import (
    load_free,
    load_hard,
    load_images,
    spearmanr,
    similarity_function,
    load_custom_embeddings_with_cache,
)

sts_dataset_fns = {"CNA": load_free, "SVOB_IMG": load_images, "SVOB_HL": load_hard}
POOLING_MODES = ["cls", "mean", "max"]


class STSEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, data, embedding_dict):
        self.emb_dict = embedding_dict
        data = self._filter_data(data)
        self.labels = torch.tensor(data["label"].to_list())
        self.embeddings1 = self._encode(data["sentence1"].to_list())
        self.embeddings2 = self._encode(data["sentence2"].to_list())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings1": self.embeddings1[idx],
            "embeddings2": self.embeddings2[idx],
            "labels": self.labels[idx],
        }

    def _filter_data(self, data):
        df = data[
            (data["sentence1"].apply(lambda x: x in self.emb_dict))
            & (data["sentence2"].apply(lambda x: x in self.emb_dict))
        ]
        if len(data) - len(df) > 0:
            print(
                f"WARNING: Embedding not found for {len(data) - len(df)}/{len(data)} samples. Ignoring."
            )
        return df

    def _encode(self, texts):
        return [self.emb_dict[text] for text in texts]


def evaluate_model_on_sts(model, main_similarity, debug=False):
    results = {}
    for name, data_fn in sts_dataset_fns.items():
        data = data_fn()
        if debug:
            data = data.iloc[:100]
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=data["sentence1"],
            sentences2=data["sentence2"],
            scores=data["label"],
            batch_size=32,
            main_similarity=main_similarity,
        )
        results[name] = evaluator(model)

    return results


def evaluate_embeddings_on_sts(embedding_dict, main_similarity, debug=False):
    results = {}
    for name, data_fn in sts_dataset_fns.items():
        data = data_fn()
        if debug:
            data = data.iloc[:100]

        dataset = STSEmbeddingDataset(data, embedding_dict)
        embeddings1 = torch.stack([item["embeddings1"] for item in dataset])
        embeddings2 = torch.stack([item["embeddings2"] for item in dataset])
        labels = [item["labels"] for item in dataset]
        similarities = similarity_function(embeddings1, embeddings2, main_similarity)
        results[name], _ = spearmanr(
            np.array([s for s in similarities]), np.array([l for l in labels])
        )
    return results


def main(args):

    if args.model_path is None and not args.eval_embeddings:
        raise ValueError(
            "Either the model path or the eval_embeddings flag must be specified."
        )

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    results = defaultdict(dict)

    pooling_modes = POOLING_MODES
    if args.eval_embeddings:
        pooling_modes = ["embeddings"]

    for pooling_mode in tqdm(pooling_modes, desc=""):

        if not args.eval_embeddings:
            word_embedding_model = models.Transformer(
                args.model_path,
                max_seq_length=128,
                tokenizer_name_or_path=args.tokenizer_path,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=pooling_mode,
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], device=0
            )

        for similarity in SimilarityFunction:

            if args.eval_embeddings:
                embeddings_dict = load_custom_embeddings_with_cache(
                    args.eval_embeddings, args.debug
                )
                results[pooling_mode][similarity] = evaluate_embeddings_on_sts(
                    embeddings_dict, main_similarity=similarity, debug=args.debug
                )

            else:
                results[pooling_mode][similarity] = evaluate_model_on_sts(
                    model, main_similarity=similarity, debug=args.debug
                )

        df = pd.DataFrame(results[pooling_mode])
        average = df.mean().to_frame("average").T
        df = pd.concat([df, average])
        results[pooling_mode] = {
            "data": df,
            "best_average": float(df.loc["average"].max()),
            "best_similarity": df.columns[df.loc["average"].argmax()],
        }

    # find the best combination of pooling and similarity
    best_pooling_mode, best_similarity, best_average = None, None, -1
    for pooling_mode in pooling_modes:
        result = results[pooling_mode]
        if result["best_average"] > best_average:
            best_pooling_mode, best_similarity, best_average = (
                pooling_mode,
                result["best_similarity"],
                result["best_average"],
            )
    results["final_result"] = {
        "pooling": best_pooling_mode,
        "similarity": best_similarity,
        "average": best_average,
    }
    print_results(results)
    return results


def print_results(results, file=None):
    print("*" * 50, "STS", "*" * 50, file=file)
    for pooling_mode in list(results.keys()):
        if pooling_mode == "final_result":
            continue
        result = results[pooling_mode]
        print(f'Pooling: "{pooling_mode}":', file=file)
        print(result["data"].round(4) * 100, file=file)
        print(f"\tBest average: {result['best_average'] * 100:.2f}", file=file)
        print(f"\tBest similarity: {result['best_similarity']}", file=file)
        print("-" * 50, file=file)
    print(file=file)
    print(f"FINAL SCORE: {results['final_result']['average'] * 100:.2f}", file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("--eval_embeddings", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
