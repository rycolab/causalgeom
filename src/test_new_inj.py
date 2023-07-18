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
#from data.embed_wordlists.embedder import load_concept_token_lists
from evals.kl_eval import load_run_Ps, load_run_output, load_model_eval,\
    correct_flag, highest_rank, highest_concept

from data.filter_generations import load_filtered_hs_wff
from evals.eval_run import filter_hs_w_ys, sample_filtered_hs

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
model_name = "gpt2-large"
concept = "number"
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
nsamples = 400

#%%

run_path = os.path.join(OUT, "run_output/number/gpt2-large/230627/run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")

run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl = load_model_eval(model_name, concept)

#%%
l0_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
)
l1_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
)

#%%
train_share = 0.5
l0_nobs = len(l0_hs_wff)
train_l0_hs, test_l0_hs = l0_hs_wff[:int(l0_nobs*train_share)], l0_hs_wff[int(l0_nobs*train_share):]

l1_nobs = len(l1_hs_wff)
train_l1_hs, test_l1_hs = l1_hs_wff[:int(l1_nobs*train_share)], l1_hs_wff[int(l1_nobs*train_share):]

#%%
l0_hs_proj = []
for h, _, _ in train_l0_hs:
#h,_,_ = l0_hs_wff[0]
    l0_hs_proj.append(h.T @ P)

l1_hs_proj = []
for h, _, _ in train_l1_hs:
#h,_,_ = l0_hs_wff[0]
    l1_hs_proj.append(h.T @ P)

#%%
def compute_avg_int(h_erase, hs_proj, nsamples, V):
    all_probs = []
    for h_proj in sample(hs_proj, nsamples):
        #h_sg = hs_proj[0]
        logits = V @ (h_erase + h_proj)
        probs = softmax(logits)
        all_probs.append(probs)
    return np.mean(np.vstack(all_probs), axis=0)
 
def score_post_int(base_distrib, l0_probs, l1_probs, faid, foid, case):
    if case == 0:
        l0id, l1id = faid, foid
    else:
        l0id, l1id = foid, faid
    return dict(
        base_correct = correct_flag(base_distrib[faid], base_distrib[foid]),
        base_correct_highest = highest_rank(base_distrib, faid),
        base_correct_highest_concept = highest_concept(base_distrib, faid, l0_tl, l1_tl),
        inj0_correct = correct_flag(l0_probs[l0id], l0_probs[l1id]),
        inj0_l0_highest = highest_rank(l0_probs, l0id),
        #inj0_l1_highest = highest_rank(l0_probs, l1id),
        inj0_l0_highest_concept = highest_concept(l0_probs, l0id, l0_tl, l1_tl),
        #inj0_l1_highest_concept = highest_concept(l0_probs, l1id, l0_tl, l1_tl),
        inj1_correct = correct_flag(l1_probs[l1id], l1_probs[l0id]),
        #inj1_l0_highest = highest_rank(l1_probs, l0id),
        inj1_l1_highest = highest_rank(l1_probs, l1id),
        #inj1_l0_highest_concept = highest_concept(l1_probs, l0id, l0_tl, l1_tl),
        inj1_l1_highest_concept = highest_concept(l1_probs, l1id, l0_tl, l1_tl),
    )


# %%
from scipy.special import softmax
from random import sample


def intervene_test_set(test_hs, case, l0_dev_hs, l1_dev_hs, V, nsamples):
    scores = []
    for h, faid, foid in tqdm(test_hs):
        #h, faid, foid = test_hs[0]
        base_distrib = softmax(V @ h)
        h_erase = h.T @ I_P
        l0_probs = compute_avg_int(h_erase, l0_dev_hs, nsamples, V)
        l1_probs = compute_avg_int(h_erase, l1_dev_hs, nsamples, V)
        score = score_post_int(
            base_distrib, l0_probs, l1_probs, faid, foid, case
        )
        scores.append(score)
    return scores
    
scores_l0 = intervene_test_set(test_l0_hs[:50], 0, l0_hs_proj, l1_hs_proj, V, 10)
scores_l1 = intervene_test_set(test_l1_hs[:50], 1, l0_hs_proj, l1_hs_proj, V, 10)

scores = scores_l0 + scores_l1
pd.DataFrame(scores).mean()

#%%
    
        #h_sg = l0_hs_proj[0]
    #    l0_logits = V @ (h_erase + h_l0)
    #    l0_probs = softmax(l0_logits)
    #    all_l0_probs.append(l0_probs)
    #for h_l1 in tqdm(sample(l1_hs_proj, nsamples)):
        #h_sg = l1_hs_proj[0]
    #    l1_logits = V @ (h_erase + h_l1)
    #    l1_probs = softmax(l1_logits)
    #    all_l1_probs.append(l1_probs)
    #all_l0_probs = np.vstack(all_l0_probs)
    #all_l1_probs = np.vstack(all_l1_probs)
    #l0_probs = np.mean(all_l0_probs, axis=0)
    #l1_probs = np.mean(all_l1_probs, axis=0)
    
# %%



# %%
