#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
#import torch
import random 
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import trange

sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.kl_eval import load_model_eval
#from data.filter_generations import load_filtered_hs_wff
#from evals.run_eval import filter_hs_w_ys, sample_filtered_hs
#from evals.run_int import get_hs_proj

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%% LOADER to be imported
def load_p_x(model_name, nucleus):
    if nucleus:
        fpath = os.path.join(DATASETS, f"p_x/{model_name}_nucleus_p_x.pkl")
    else:
        fpath = os.path.join(DATASETS, f"p_x/{model_name}_p_x.pkl")
    
    with open(fpath, "rb") as f:
        p_x = pickle.load(f)
    return p_x

#%% Helpers
def load_format_counts(x_counts_path):
    with open(x_counts_path, "rb") as f:
        x_counts = pickle.load(f)
    x_counts_list = []
    for k, v in x_counts.items():
        x_counts_list.append((k,v))
    xdf = pd.DataFrame(x_counts_list, columns=["token", "count"])
    return xdf

def get_p_x(counts_df, vocab_size):
    vocab_ref = pd.DataFrame([x for x in range(vocab_size)], columns=["token"])

    token_counts = pd.merge(
        left=vocab_ref,
        right=counts_df,
        on="token",
        how="outer"
    )

    token_counts["count"].fillna(value=0, inplace=True)
    token_counts["padded_count"] = token_counts["count"] + 1
    token_counts["p_x"] = token_counts["padded_count"] / token_counts["padded_count"].sum()
    token_counts.sort_values(by = "token", inplace=True)
    p_x = token_counts["p_x"]
    #h_x = entropy(p_x)
    return p_x#, h_x

#%%
if __name__=="__main__": 
    ####PARAMS
    #model_name = "gpt2-base-french"
    model_name = "gpt2-large"
    #concept = "gender"
    concept = "number"
    nucleus = True

    #### RUNNER
    #TODO: GET RID OF THIS DISTINCTION 
    if nucleus:
        counts_path = os.path.join(OUT, f"p_x/{model_name}/x_counts_{model_name}_nucleus.pkl")
    elif model_name == "gpt2-large" and not nucleus:
        counts_path = os.path.join(OUT, f"p_x/{model_name}/x_counts_{model_name}.pkl")
    elif model_name == "gpt2-base-french" and not nucleus:
        counts_path = os.path.join(OUT, f"p_x/{model_name}/x_counts_{model_name}_no_I_P.pkl")
    else:
        raise ValueError("Wrong model nucleus combo")

    outdir = os.path.join(DATASETS, f"p_x")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if nucleus:
        outpath = os.path.join(outdir, f"{model_name}_nucleus_p_x.pkl")
    else:
        outpath = os.path.join(outdir, f"{model_name}_p_x.pkl")

    V, _, _ = load_model_eval(model_name, concept)
    counts_df = load_format_counts(counts_path)
    p_x = get_p_x(counts_df, V.shape[0])

    with open(outpath, "wb") as f:
        pickle.dump(p_x, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported p_x for {model_name}, nuc {nucleus}, ent: {entropy(p_x)}")

# %%
