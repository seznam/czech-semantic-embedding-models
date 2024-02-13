import pickle
from pathlib import Path
from sentence_transformers.evaluation import SimilarityFunction
from scipy.stats import spearmanr
from typing import Literal
from costra import costra
from collections import defaultdict, Counter

import pandas as pd
import torch
import requests
import tarfile
import zipfile
import os


def load_hard():
    data_file = "data/hard.tsv"
    data_file_result = "data/hard-result.tsv"
    dtype = {"sentence1": str, "sentence2": str, "label": float}
    data = pd.read_csv(
        data_file, delimiter="\t", names=["sentence1", "sentence2"], dtype=dtype
    )
    labels = pd.read_csv(data_file_result, names=["label"], dtype=dtype)
    data["label"] = labels
    return data


def load_images():
    data_file = "data/images.tsv"
    data_file_result = "data/images-result.tsv"
    dtype = {"sentence1": str, "sentence2": str, "label": float}
    data = pd.read_csv(
        data_file, delimiter="\t", names=["sentence1", "sentence2"], dtype=dtype
    )
    labels = pd.read_csv(data_file_result, names=["label"], dtype=dtype)
    data["label"] = labels
    return data


def load_free():
    data_file = "data/free-test.tsv"
    dtype = {"sentence1": str, "sentence2": str, "label": float}
    data = pd.read_csv(
        data_file,
        delimiter="\t",
        names=["sentence1", "sentence2", "label", "annotations", "median"],
        quoting=3,
        dtype=dtype,
    )
    data = data[["sentence1", "sentence2", "label"]]
    return data


def load_costra():
    return costra.get_sentences()


def load_facebook():
    posts_file_name = "data/facebook/gold-posts.txt"
    labels_file_name = "data/facebook/gold-labels.txt"

    if not Path(posts_file_name).exists() or not Path(labels_file_name).exists():
        url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0022-FE82-7/facebook.zip"
        filename = "data/facebook.zip"
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            raise RuntimeError("Could not download CFD dataset.")

        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path="data/")

        os.remove(filename)

    posts = pd.read_csv(posts_file_name, names=["post"], sep="~")
    labels = pd.read_csv(labels_file_name, names=["label"], sep="~")

    df = pd.concat([posts, labels], axis=1)
    df = df[df["label"] != "b"]
    label_map = {"n": 0, "0": 1, "p": 2}
    df["label"] = df["label"].map(label_map)

    return df


def load_dareczech(
    t: Literal["train", "test", "dev"], debug=False, n_data=None, random_state=42
):
    DATASET_NAME = {"train": "train_big.tsv", "test": "test.tsv", "dev": "dev.tsv"}
    dataset_path = Path("data/dareczech") / DATASET_NAME[t]

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            "DareCzech dataset is not downloaded, please read the information in data/README.md"
        )

    df = pd.read_csv(dataset_path, sep="\t", quoting=3, nrows=1000 if debug else None)
    if not debug and n_data is not None:
        df = df.sample(n_data, random_state=random_state)
    return df


def load_ctdc():
    d = Path("data/czech_text_document_corpus_v20")
    if not d.exists():
        # download data
        url = "http://ctdc.kiv.zcu.cz/czech_text_document_corpus_v20.tgzm"
        output_file = "data/czech_text_document_corpus_v20.tgz"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.raw.read())
        else:
            raise RuntimeError(
                f"Could not download CTDC dataset. - see data/README.md. {response.content}"
            )

        with tarfile.open(output_file, "r") as tar:
            tar.extractall(path="data/")

    # parse data
    cat_counter = Counter()
    for file in d.glob("*.txt"):
        cats = file.name.split(".")[0].split("_")[1:]
        cat_counter.update(cats)

    top_cats = [cat[0] for cat in cat_counter.most_common(37)]
    rows = []
    for file in d.glob("*.txt"):
        with open(file, "rt") as f:
            text = f.readline()
        cats = set(file.name.split(".")[0].split("_")[1:])
        row_cats = {cat: 0 for cat in top_cats}
        for cat in cats:
            if cat in row_cats:
                row_cats[cat] += 1

        # add text
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        row_cats["text"] = text

        rows.append(row_cats)

    df = pd.DataFrame(rows)
    return df


def load_all_texts():
    texts = []

    for data_fn in [load_hard, load_images, load_free]:
        data = data_fn()
        texts.extend(data["sentence1"].to_list())
        texts.extend(data["sentence2"].to_list())

    texts.extend(load_costra())
    texts.extend(load_ctdc()["text"])
    texts.extend(load_facebook()["post"])

    try:
        for t in ["dev", "test"]:
            data = load_dareczech(t)
            texts.extend(data["doc"].tolist())
            texts.extend(data["query"].unique().tolist())
    except FileNotFoundError as e:
        print(e)
        print("WARNING: DareCzech not found, skipping.")

    return texts


def similarity_function(tensor1, tensor2, method):
    if method == "cs" or method == SimilarityFunction.COSINE:
        return torch.cosine_similarity(tensor1, tensor2, dim=1)
    if method == "ed" or method == SimilarityFunction.EUCLIDEAN:
        a_norm = torch.nn.functional.normalize(tensor1, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(tensor2, p=2, dim=1)
        return 1 - (a_norm - b_norm).pow(2).sum(1).sqrt()
    if method == "dot" or method == SimilarityFunction.DOT_PRODUCT:
        return torch.tensor([torch.dot(a, b) for a, b in zip(tensor1, tensor2)])
    if method == "man" or method == SimilarityFunction.MANHATTAN:
        a_norm = torch.nn.functional.normalize(tensor1, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(tensor2, p=2, dim=1)
        return 1 - torch.abs(a_norm - b_norm).sum(1)
    raise NotImplementedError("")


def load_custom_embeddings(path, debug=False):
    if debug:
        all_texts = load_all_texts()
        embeddings = torch.rand(len(all_texts), 256).float()
        return {k: e for k, e in zip(all_texts, embeddings)}

    with open(path, "rb") as f:
        return pickle.load(f)


embeddings_cache = [None]


def load_custom_embeddings_with_cache(path=None, debug=False):
    embeddings_cache[0] = (
        load_custom_embeddings(path, debug)
        if embeddings_cache[0] is None
        else embeddings_cache[0]
    )
    return embeddings_cache[0]


class EmbeddingForSequenceClassification(torch.nn.Module):
    def __init__(
        self, hidden_size, loss=torch.nn.CrossEntropyLoss(), num_labels=3, dropout=0.1
    ):
        super().__init__()
        self.classifier_dense = torch.nn.Linear(hidden_size, hidden_size)
        self.classifier_dropout = torch.nn.Dropout(dropout)
        self.classifier_out_proj = torch.nn.Linear(hidden_size, num_labels)
        self.classifier_activation = torch.nn.GELU()
        self.loss = loss

    def _get_logits(self, embeddings):
        x = embeddings
        x = self.classifier_dropout(x)
        x = self.classifier_dense(x)
        x = self.classifier_activation(x)
        x = self.classifier_dropout(x)
        x = self.classifier_out_proj(x)
        return x

    def forward(self, embeddings, labels=None):
        logits = self._get_logits(embeddings)

        if labels is not None:
            loss = self.loss(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


if __name__ == "__main__":
    pass
