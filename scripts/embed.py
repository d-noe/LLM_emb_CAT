import pandas as pd
import numpy as np
import argparse
import os
import pathlib
from tqdm import tqdm
from datasets import load_dataset

from sentence_transformers import SentenceTransformer

def compute_save_embs(
    model_emb,
    df=pd.DataFrame,
    base_path:str=None,
    columns_to_embed:list=None,
    columns_suffixes:list=None,
    do_save:bool=True
):
    assert (not (do_save and (base_path is None))), "`do_save` set to `True` but no `base_path` provided."

    if columns_to_embed is None:
        model_names = [col.split("_ans")[0] for col in df.columns if "_answer" in col]
        columns_to_embed = ["question"]+model_names
        columns_suffixes = [None]+["_answers"]*len(model_names)
    if columns_suffixes is None:
        columns_suffixes = [None]*len(columns_to_embed)

    dict_embs = {}
    for c, c_s in tqdm(zip(columns_to_embed, columns_suffixes)):
        if not c_s is None:
            curr_col = c+c_s
        else:
            curr_col = c

        try:
            loaded_embs = np.load(f"{base_path}/{c}.npy")
            if len(loaded_embs)==len(df): # minimal check /!\ use cautiously
                embs = loaded_embs
            else:
                embs = model_emb.encode([a if type(a)==str else a[0] for a in df[curr_col].to_list()])
        except:
            embs = model_emb.encode([a if type(a)==str else a[0] for a in df[curr_col].to_list()])
            
        if do_save:
            np.save(f"{base_path}/{c}.npy", embs)

        dict_embs[c] = embs
    
    return dict_embs

if __name__=="__main__":
    # 0. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", "-i",
        help="Path to the input .csv file (must have columns 'question' and '<MODEL>_answers').",
        default="noepsl/H-LLMC2"
    )
    parser.add_argument(
        "--output_folder", "-o",
        help="Path to the output folder. Embedding will be saved at ./<OUTPUT_FOLDER>/<ENCODER_NAME>/<COL_REF>.npy",
        default="data/hc3_data/embs/"
    )
    parser.add_argument(
        "--encoder_name", "-e",
        help="Encoder name. Must be compliant with `sentence_transformers` library (i.e., SentenceTransformer(<ENCODER_NAME>) ).",
        default="avsolatorio/GIST-Embedding-v0"
    )
    parser.add_argument(
        "--columns", "-c",
        nargs='+', 
        default=None,
        help="Columns to be embedded."
    )
    args = parser.parse_args()

    # 1. Load Data
    # .csv with columns "question" and "<MODEL>_answers" for each model
    try:
        data = load_dataset(args.input_path)
        df = data["train"].to_pandas()
    except:
        df = pd.read_csv(args.input_path)
    df = df.fillna('')

    # 2. Load Encoder
    model = SentenceTransformer(args.encoder_name)

    # 3. Embed & save
    model_name = args.encoder_name.split("/")[1] if "/" in args.encoder_name else args.encoder_name
    save_folder = os.path.join(args.output_folder,model_name)
    
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True) 

    compute_save_embs(
        model_emb=model,
        base_path=save_folder,
        df=df,
        columns_to_embed=args.columns,
        columns_suffixes=None,
        do_save=True
    )