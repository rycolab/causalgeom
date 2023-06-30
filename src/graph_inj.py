#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
import csv

import numpy as np
import pandas as pd
import pickle
#import torch
import random 
from tqdm import trange, tqdm

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, \
    get_concept_name
from data.embed_wordlists.embedder import load_concept_token_lists
from evals.kl_eval import load_run_Ps, load_run_output, \
    compute_eval_filtered_hs, load_model_eval, compute_kl, \
        renormalize, get_distribs, correct_flag, highest_rank, highest_concept
from analysis.format_res import get_best_runs
from data.filter_generations import load_filtered_hs, load_filtered_hs_wff
from test_eval import filter_test_hs_wff, create_er_df, create_fth_df, \
    compute_kl_baseline

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
folderpath = os.path.join(RESULTS, "inj")
files = os.listdir(folderpath)

dfs = []
for fpath in files:
    path = os.path.join(folderpath, fpath)
    df = pd.read_csv(path, index_col=0).reset_index(names="metric")
    #df.columms = ["metric", "l0_means", "l1_means", "all_means"]
    splitpath = fpath.split("_")
    concept, model = splitpath[0], splitpath[1]
    df["concept"] = concept
    df["model"] = model
    dfs.append(df)


# %%
full_df = pd.concat(dfs, axis=0)
mean_df = full_df.groupby(["model", "concept", "metric"]).mean().reset_index()
# %%
test = mean_df[mean_df["model"] == "gpt2-large"].sort_values(by="metric")
# %%
test[test["metric"].isin([
    "base_correct", "I_P_correct", "I_P_inj0_correct", 
    "I_P_inj0_l0_highest", "I_P_inj0_l0_highest_concept",
    "I_P_inj1_correct", "I_P_inj1_l1_highest", 
    "I_P_inj1_l1_highest_concept"])]