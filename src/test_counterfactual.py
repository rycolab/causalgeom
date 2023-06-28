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
nsamples = 50

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

# %%
V0 = V[l0_tl]
V1 = V[l1_tl]

V0proj = V0 @ P
V1proj = V1 @ P

V0_mean = np.mean(V0proj, axis=0)
V1_mean = np.mean(V1proj, axis=0)

V0proj_norm = np.linalg.norm(V0proj, axis=0)
V1proj_norm = np.linalg.norm(V1proj, axis=0)
V0proj_normed = V0proj * V0proj_norm
V0_meannorm = 

V0meannorm = V0_mean / np.linalg.norm(V0_mean)
V1meannorm = V1_mean / np.linalg.norm(V1_mean)

#%%
#from sklearn.metrics.pairwise import cosine_similarity
#from scipy import sparse
#from scipy.spatial.distance import cdist
#import seaborn as sns
#A =  np.vstack([V0[:10], V1[:10], V0_mean.reshape(1,-1), V1_mean.reshape(1,-1)])
#sns.heatmap(cdist(A, A, metric = "cosine"))
#%%
#h, faid, foid = test_l0_hs_wff[0]

#%%
from evals.kl_eval import get_distribs, correct_flag
from tqdm import tqdm

alpha = 5

l0_Phn = []
l0_reslist = []
for h, faid, foid in tqdm(test_l0_hs_wff):
    #h, faid, foid = test_l0_hs_wff[0]
    base_distribs = get_distribs(h, V, l0_tl, l1_tl)
    I_P_distribs = get_distribs(h.T @ I_P, V, l0_tl, l1_tl)
    normPh = np.linalg.norm(h.T @ P)
    l0_Phn.append(normPh)
    I_P_inj0_distribs = get_distribs((h.T @ I_P) + (V0meannorm * alpha * normPh), V, l0_tl, l1_tl)
    I_P_inj1_distribs = get_distribs((h.T @ I_P) + (V1meannorm * alpha * normPh), V, l0_tl, l1_tl)

    l0_reslist.append(dict(
        base_correct = correct_flag(base_distribs["all_split"][faid], base_distribs["all_split"][foid]),
        I_P_correct = correct_flag(I_P_distribs["all_split"][faid], I_P_distribs["all_split"][foid]),
        I_P_inj0_correct = correct_flag(I_P_inj0_distribs["all_split"][faid], I_P_inj0_distribs["all_split"][foid]),
        I_P_inj1_correct = correct_flag(I_P_inj1_distribs["all_split"][foid], I_P_inj1_distribs["all_split"][faid]),
    ))

l1_Phn = []
l1_reslist = []
for h, faid, foid in tqdm(test_l1_hs_wff):
    base_distribs = get_distribs(h, V, l0_tl, l1_tl)
    I_P_distribs = get_distribs(h.T @ I_P, V, l0_tl, l1_tl)
    normPh = np.linalg.norm(h.T @ P)
    l1_Phn.append(normPh)
    I_P_inj0_distribs = get_distribs((h.T @ I_P) + (V0meannorm * alpha * normPh), V, l0_tl, l1_tl)
    I_P_inj1_distribs = get_distribs((h.T @ I_P) + (V1meannorm * alpha * normPh), V, l0_tl, l1_tl)

    l1_reslist.append(dict(
        base_correct = correct_flag(base_distribs["all_split"][faid], base_distribs["all_split"][foid]),
        I_P_correct = correct_flag(I_P_distribs["all_split"][faid], I_P_distribs["all_split"][foid]),
        I_P_inj0_correct = correct_flag(I_P_inj0_distribs["all_split"][foid], I_P_inj0_distribs["all_split"][faid]),
        I_P_inj1_correct = correct_flag(I_P_inj1_distribs["all_split"][faid], I_P_inj1_distribs["all_split"][foid]),
    ))


# %%
print("Results for singular (0) contexts:")
print(pd.DataFrame(l0_reslist).mean())
print("Results for plural (1) contexts:")
print(pd.DataFrame(l1_reslist).mean())
print("Results for all contexts:")
print(pd.DataFrame(l0_reslist + l1_reslist).mean())

# %%
