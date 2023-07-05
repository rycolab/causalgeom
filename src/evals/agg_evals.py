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
#from scipy.special import softmax
#from scipy.stats import entropy
from tqdm import trange

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
#from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
#from data.embed_wordlists.embedder import load_concept_token_lists
#from evals.kl_eval import load_run_Ps, load_run_output, \
#    compute_eval_filtered_hs, load_model_eval, compute_kl, \
#        renormalize, get_distribs

#from analysis.format_res import get_best_runs
#from data.filter_generations import load_filtered_hs, load_filtered_hs_wff

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
def create_agg_dfs(pairs):
    acc_dfs = []
    fth_dfs = []
    er_dfs = []
    for model_name, concept in pairs:
        eval_dir = os.path.join(RESULTS, f"{concept}/{model_name}")
        run_evals = [x for x in os.listdir(eval_dir) if x.endswith(".pkl")]

        for run_eval in run_evals:
            run_eval_path = os.path.join(eval_dir, run_eval)
            with open(run_eval_path, 'rb') as f:      
                run_eval = pickle.load(f)
            acc_dfs.append(run_eval["acc_df"])
            fth_dfs.append(run_eval["fth_df"])
            er_dfs.append(run_eval["er_df"])
    return acc_dfs, fth_dfs, er_dfs

#%%#################
# Main             #
####################
if __name__=="__main__":
    #args = get_args()
    #logging.info(args)

    agg_pairs = [
        ("gpt2-large", "number"),
        ("bert-base-uncased", "number"),
        ("gpt2-base-french", "gender"),
        ("camembert-base", "gender"),
    ]
    all_acc_dfs, all_fth_dfs, all_er_dfs = create_agg_dfs(agg_pairs)

    outdir = RESULTS
    all_acc_df = pd.concat(all_acc_dfs,axis=0)
    all_fth_df = pd.concat(all_fth_dfs,axis=0)
    all_er_df = pd.concat(all_er_dfs,axis=0)
    all_acc_df.to_csv(os.path.join(outdir, f"acc.csv"), index=False)
    all_fth_df.to_csv(os.path.join(outdir, f"fth.csv"), index=False)
    all_er_df.to_csv(os.path.join(outdir, f"er.csv"), index=False)
    logging.info("Finished exporting all results.")

