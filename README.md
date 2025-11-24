# LLM Communication Accommodation 🗣️

## Overview

This is the companion repository to the paper *LLMs Stick to the Point, Humans to Style: Semantic and Stylistic Alignment in Human and LLM Communication* (citation details coming soon).

The code provided here is used to analyze question-answering mechanisms in human-LLM communication (and to compare it against huùan-human dialogues), based on the encoder-based vector representation spaces.

## Installation

To set up the project environment, clone or download the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Reproduction

To reproduce the results, run the shell script run_scripts.sh from the root directory. This will execute the embedding and similarity scripts.

```bash
cd scripts
./run_scripts.sh
```

The pipeline will:
1. Load dataset and compute embeddings ([`embed.py`](./scripts/embed.py)). By default, the embeddings will be stored in the `data` folder (`./data/<dataset_name>/embs/<encoder_name>`).
2. Compute QA-pairs similarities based on pre-computed vector representations ([`compute_sims.py`](./scripts/compute_sims.py)). By default, similarity scores will be stored in the `data` folder (`./data/<dataset_name>/sims/<encoder_name>.csv`).

After the script is finished, open the [`results.ipynb`](./results.ipynb) Jupyter Notebook to view and generate the plots (Fig. 1).


## Citation

This code was originally produced for the paper [*LLMs stick to the point, humans to style: Semantic and Stylistic Alignment in Human and LLM Communication*](https://aclanthology.org/2025.sigdial-1.16/) (Durandard et al., SIGDIAL 2025).

```
@inproceedings{durandard-etal-2025-llms,
    title = "{LLM}s stick to the point, humans to style: Semantic and Stylistic Alignment in Human and {LLM} Communication",
    author = "Durandard, No{\'e}  and
      Dhawan, Saurabh  and
      Poibeau, Thierry",
    editor = "B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Lef{\`e}vre, Fabrice  and
      Asher, Nicholas  and
      Kim, Seokhwan  and
      Merlin, Teva",
    booktitle = "Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = aug,
    year = "2025",
    address = "Avignon, France",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.sigdial-1.16/",
    pages = "206--213"
}
```
