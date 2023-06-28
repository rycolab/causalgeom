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

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
from data.embed_wordlists.embedder import load_concept_token_lists
from evals.kl_eval import load_run_Ps, load_run_output, \
    compute_eval_filtered_hs, load_model_eval, compute_kl, \
        renormalize, get_distribs

from analysis.format_res import get_best_runs
from data.filter_generations import load_filtered_hs, load_filtered_hs_wff


coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
model_name = "gpt2-large"
concept = "number"
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
nsamples = 200

#%%
from test_eval import filter_test_hs_wff, create_er_df, create_fth_df, compute_kl_baseline
run_path = os.path.join(OUT, "run_output/number/gpt2-large/230627/run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")
run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl, _ = load_model_eval(model_name, concept)
#l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
    run["X_test"], run["facts_test"], run["foils_test"], 
    l0_tl, l1_tl, nsamples=nsamples
)
test_eval_1 = compute_eval_filtered_hs(
    model_name, concept, P, I_P, test_l0_hs_wff, test_l1_hs_wff
)
test_kl_baseline_1 = compute_kl_baseline(
    run["X_test"], V, l0_tl, l1_tl, nsamples=nsamples
)

fth_df_1 = create_fth_df(
    test_eval_1, None, test_kl_baseline_1, None, concept, 
    model_name, " ", run["config"]["k"]
)
# %%
run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl, _ = load_model_eval(model_name, concept)
#l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
    run["X_test"], run["facts_test"], run["foils_test"], 
    l0_tl, l1_tl, nsamples=nsamples
)
test_eval_2 = compute_eval_filtered_hs(
    model_name, concept, P, I_P, test_l0_hs_wff, test_l1_hs_wff
)
test_kl_baseline_2 = compute_kl_baseline(
    run["X_test"], V, l0_tl, l1_tl, nsamples=nsamples
)

fth_df_2 = create_fth_df(
    test_eval_2, None, test_kl_baseline_2, None, concept, 
    model_name, " ", run["config"]["k"]
)
# %%
pd.concat([fth_df_1, fth_df_2], axis=0)
#%%
fpath = os.path.join(RESULTS,"gender/gpt2-base-french/eval_run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
with open(fpath, "rb") as f:
    exported_df = pickle.load(f)

# %%

