import json
import pandas as pd
import torch
import pickle
from utils import load_all_texts


def save_openada_embeddings():
    df = pd.read_csv("data/openai_embeddings.tsv", sep="\t", quoting=3)
    embeddings_dict = {
        text: torch.tensor(json.loads(emb))
        for text, emb in zip(df["doc"], df["embedding"])
    }
    with open("data/openai_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)


def save_custom_embeddings():

    # loads every sentence from all the datasets
    texts = load_all_texts()
    embeddings = ...  # process text into custom embeddings
    embeddings_dict = {k: e for k, e in zip(texts, embeddings)}
    with open("data/openai_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings_dict, f)


if __name__ == "__main__":
    save_openada_embeddings()
