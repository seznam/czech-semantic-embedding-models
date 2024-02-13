import argparse
import numpy as np

from costra import costra
from sentence_transformers import SentenceTransformer, models
import torch
from utils import load_custom_embeddings_with_cache, load_costra
import io
from contextlib import redirect_stdout


class CostraEmbeddingDataset(torch.utils.data.Dataset):
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


def load_model(model_path, device=0, pooling_mode="cls", max_seq_length=128):
    word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model], device=device
    )
    return model


def main(args):

    if args.eval_embeddings:
        emb_dict = load_custom_embeddings_with_cache(args.eval_embeddings, args.debug)
        sentences = load_costra()
        dataset = CostraEmbeddingDataset(sentences, emb_dict)
        embeddings = np.array([item["embeddings"].numpy() for item in dataset])
    else:
        if args.tokenizer_path is None:
            args.tokenizer_path = args.model_path

        sentences = costra.get_sentences()
        model = load_model(args.model_path, 0, args.pooling, max_seq_length=128)
        embeddings = np.array(model.encode(sentences))

    f = io.StringIO()
    with redirect_stdout(f):
        costra.evaluate(embeddings)

    results = f.getvalue()
    print_results(results)
    return results


def print_results(results, file=None):
    print("*" * 50, "COSTRA", "*" * 50, file=file)
    print(results, file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument(
        "-p",
        "--pooling",
        type=str,
        default="cls",
        help="one in ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']",
    )
    parser.add_argument("--eval_embeddings", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
