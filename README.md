# Czech Sentence Embedding Models

This repository contains pre-trained small-sized **Czech** sentence embedding models developed and evaluated as part of our research paper [Some Like It Small: Czech Semantic Embedding Models for Industry Applications](#Acknowledgements)


## Repository Contents

- [Released models](#Released-Models)
- [Evaluation results](#Evaluation-Results)
- [Evaluation scripts](#Evaluation-Pipeline)

## Released Models

Our models were trained to generate high-quality sentence embeddings, which can be applied to a range of natural language processing tasks such as similarity search, retrieval, clustering or classification.

All models have 15-20M parameters, generate 256-dimensional embeddings, and process up to 128 tokens, except for RetroMAE (up to 512 tokens). They all support Czech language, and the distilled models also handle English.

You can access the models weights and configuration files through the provided Hugging Face links below. All models are available under the CC-BY4.0 LICENSE, except those trained with the Czeng dataset, which are under the CC-BY-NC 4.0 LICENSE.

| *Model*                     | HuggingFace link                                                                                            | Info                                                                                                                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RetroMAE-Small              | [Seznam/retromae-small-cs](https://huggingface.co/Seznam/retromae-small-cs)                                 | BERT-small model trained with the RetroMAE objective on a custom Czech corpus.                                                                                                                            |
| Dist-MPNet-ParaCrawl        | [Seznam/dist-mpnet-paracrawl-cs-en](https://huggingface.co/Seznam/dist-mpnet-paracrawl-cs-en)               | BERT-small model distilled from the `sentence-transformers/all-mpnet-base-v2` model, using parallel cs-en dataset ParaCrawl for training.                                                                 |
| Dist-MPNet-CzEng            | [Seznam/dist-mpnet-czeng-cs-en](https://huggingface.co/Seznam/dist-mpnet-czeng-cs-en)                       | BERT-small model BERT-small model distilled from the `sentence-transformers/all-mpnet-base-v2` model, using parallel cs-en dataset Czeng for training. Exclusively accessible under the CC-BY-NC license. |
|                             |                                                                                                             |                                                                                                                                                                                                           |
| SimCSE-RetroMAE-Small       | [Seznam/simcse-retromae-small-cs](https://huggingface.co/Seznam/simcse-retromae-small-cs)                   | The RetroMAE-Small model fine-tuned with the SimCSE.                                                                                                                                                      |
| SimCSE-Dist-MPNet-ParaCrawl | [Seznam/simcse-dist-mpnet-paracrawl-cs-en](https://huggingface.co/Seznam/simcse-dist-mpnet-paracrawl-cs-en) | The Dist-MPNet-ParaCrawl model fine-tuned with the SimCSE.                                                                                                                                                |
| SimCSE-Dist-MPNet-CzEng     | [Seznam/simcse-dist-mpnet-czeng-cs-en](https://huggingface.co/eznam/simcse-dist-mpnet-czeng-cs-en)          | The Dist-MPNet-CzEng fine-tuned with the SimCSE. Exclusively accessible under the CC-BY-NC license.                                                                                                       |
| SimCSE-Small-E-Czech        | [Seznam/simcse-small-e-czech](https://huggingface.co/Seznam/simcse-small-e-czech)                           | Czech ELECTRA model (small-e-czech) fine-tuned with the SimCSE objective to enhance sentence embeddings.                                                                                                  |

You can easily use the pre-trained models in your own applications or projects:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "Seznam/retromae-small-cs"  # link name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_texts = [
    "Dnes je výborné počasí na procházku po parku.",
    "Večer si oblíbím dobrý film a uvařím si čaj."
]

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=128, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0] # CLS

similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
```

## Evaluation Results

Intrinsic (semantic tasks) and extrinsic (real world NLP tasks), were conducted to understand each model's performance. Public Czech datasets were used for these evaluations [paper link]().

### Zero-shot

| Model                       | all STS (average pearsonr) | Costra (Acc) | DaReCzech (P@10) |
| --------------------------- | -------------------------- | ------------ | ---------------- |
| Small-E-Czech               | 48.30                      | 64.29        | 37.31            |
|                             |                            |              |                  |
| RetroMAE-Small              | 76.30                      | 69.66        | 42.16            |
| Dist-MPNet-ParaCrawl        | 84.25                      | 70.42        | 42.33            |
| Dist-MPNet-CzEng            | 87.60                      | 71.22        | 42.01            |
|                             |                            |              |                  |
| SimCSE-Small-E-Czech        | 66.24                      | 66.44        | 39.20            |
| SimCSE-RetroMAE-Small       | 78.66                      | 69.63        | 42.04            |
| SimCSE-Dist-MPNet-ParaCrawl | 85.00                      | 71.12        | ***42.38***      |
| SimCSE-Dist-MPNet-CzEng     | ***87.83***                | ***71.77***  | 42.18            |


### Linear Probing Results

| Model                       | CFD (F1 score) | CTDC (F1 score) |
| --------------------------- | -------------- | --------------- |
| Small-E-Czech               | 32.38          | 25.92           |
|                             |                |                 |
| RetroMAE-small              | 68.56          | 78.18           |
| Dist-MPNet-ParaCrawl        | 71.30          | 80.18           |
| Dist-MPNet-CzEng            | ***72.84***    | ***82.38***     |
|                             |                |                 |
| SimCSE-Small-E-Czech        | 54.78          | 50.06           |
| SimCSE-RetroMAE-small       | 68.70          | 77.32           |
| SimCSE-Dist-MPNet-ParaCrawl | 71.87          | 79.41           |
| SimCSE-Dist-MPNet-CzEng     | 72.66          | 81.63           |


### Fine-tuning Evaluation Results

| Model                       | CFD (F1 score) | CTDC (F1 score) | DaReCzech (P@10) |
| --------------------------- | -------------- | --------------- | ---------------- |
| Small-E-Czech               | 76.94          | 58.12           | 43.64            |
|                             |                |                 |                  |
| RetroMAE-Small              | 76.85          | 84.58           | 45.29            |
| Dist-MPNet-ParaCrawl        | 77.42          | 86.02           | 45.55            |
| Dist-MPNet-CzEng            | ***78.73***    | 85.85           | ***45.75***      |
|                             |                |                 |                  |
| SimCSE-Small-E-Czech        | 76.27          | 68.33           | 44.64            |
| SimCSE-RetroMAE-Small       | 76.16          | 84.95           | 45.26            |
| SimCSE-Dist-MPNet-ParaCrawl | 77.31          | ***86.10***     | 45.66            |
| SimCSE-Dist-MPNet-CzEng     | ***78.73***    | 85.25           | ***45.75***      |


## Evaluation Pipeline

Coming soon, stay tuned for updates.

## Acknowledgements

If you find our work helpful, please consider citing us:

TODO
```
@article{,
  title={Some Like It Small: Czech Semantic Embedding Models for Industry Applications},
  volume={},
  url={},
  DOI={},
  number={},
  journal={},
  author={Bednář, Jiří and Náplava, Jakub and Barančíková, Petra and Ondřej, Lisický},
  year={2024},
  month={},
  pages={}
}
```
