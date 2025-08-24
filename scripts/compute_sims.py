import pandas as pd
import numpy as np
import argparse

import os
from os import listdir
from os.path import isfile, join
import pathlib

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim

def get_sim_metric(metric_name:str):
    name2method = {
        "cosine": cos_sim,#cosine_similarity,
        "euclidean": pairwise_distances,
        "dot_product": np.inner
    }

    if metric_name in name2method.keys():
        return name2method[metric_name]
    else:
        raise NotImplementedError(f"Metric '{metric_name}' not implemented.\nImplemented methods are: {', '.join([f'`{m}`' for m in name2method.keys()])}.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", "-m",
        type=str, 
        default="cosine",
        help="Similarity (/ distance) method to use."
    )
    parser.add_argument(
        "--reference_file", "-r",
        type=str, 
        default="question",
        help="Reference column to compute similarity with."
    )
    parser.add_argument(
        "--embedding_path", "-e",
        type=str, 
        default="data/hc3_data/embs/GIST-Embedding-v0",
        help="Path to pre-computed stored embeddings."
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str, 
        default="data/hc3_data/sims/GIST-Embedding-v0_sims.csv",
        help="Path where to store computed similarities."
    )
    args = parser.parse_args()

    # 0. Prepare
    metric = get_sim_metric(args.metric)

    df_sims = pd.DataFrame()

    # 1. Retrieve files
    files = [f.replace(".npy", "") for f in listdir(args.embedding_path) if isfile(join(args.embedding_path, f)) and f.endswith(".npy")]
    assert args.reference_file in files, f"Reference embedding `{args.reference_file}` not in embedding folder `{args.embedding_path}`."
    files.remove(args.reference_file)

    # 2. Compute sims and store
    save_folder = os.path.dirname(args.output_path)
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True) 

    emb_m = args.embedding_path.split("/")[-1]

    embs_ref = np.load(f"{args.embedding_path}/{args.reference_file}.npy")
    for f in files:
        embs_f = np.load(f"{args.embedding_path}/{f}.npy")
        curr_sims = [metric(e_r,e_f).item() for e_r, e_f in zip(embs_ref, embs_f)]
        df_sims[f"{emb_m}_sim_{f}"] = curr_sims

    if not os.path.exists(directory):
        os.makedirs(directory)
    df_sims.to_csv(args.output_path, index=False)
