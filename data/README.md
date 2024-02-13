# Datasets

## STS
All STS datasets are already included in the `data` folder.
- For additional information on `hard*` and `images*`, please visit [this link](https://github.com/Svobikl/sts-czech/tree/master). (`complete_corpus` / `headlines`, `images`)
- For details on `free-test`, refer to [this paper](https://arxiv.org/abs/2108.08708).
  
## COSTRA
COSTRA dataset should be downloaded automatically.
For more information about the dataset, please visit [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3248?show=full&locale-attribute=cs) or [github](https://github.com/barancik/costra)

## CFD
The CFD dataset should be automatically downloaded upon first evaluation. If not, you can manually download and unzip it into the `data/` folder using the following commands:



```
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0022-FE82-7{/facebook.zip}
unzip file.zip
```

For more information about the dataset, please visit [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0022-FE82-7).

## CTDC
The CTDC dataset will **NOT** be downloaded automatically. You can obtain it by following the steps outlined [here](http://ctdc.kiv.zcu.cz). Once obtained, save the contents of the decompressed TGZ file as `data/czech_text_document_corpus_v20`.

## DareCzech
The DareCzech dataset will **NOT** be downloaded automatically. However, you can obtain it by following the instructions provided [here](https://github.com/seznam/DaReCzech). Save the DareCzech dataset as `data/dareczech`.