#!/bin/bash

encoder_names='avsolatorio/GIST-Embedding-v0 StyleDistance/styledistance'
metric_name='cosine'
input_data_path='noepsl/H-LLMC2'
base_save_path='../data/hllmc2'

for encoder in $encoder_names
do
    # embed questions and answers
    echo $encoder
    python embed.py -i "$input_data_path" -o "${base_save_path}/embs/" -e "$encoder"

    encoder_base="${encoder##*/}"
    emb_path="${base_save_path}/embs/${encoder_base}"
    sim_path="${base_save_path}/sims/${encoder_base}.csv"
    # compute pairwise similarities between question and answers
    python compute_sims.py -m "$metric_name" -r "question" -e "$emb_path" -o "$sim_path"
done
