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

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.kl_eval import load_run_Ps, load_run_output, load_model_eval
from data.filter_generations import load_filtered_hs_wff
from evals.run_eval import filter_hs_w_ys, sample_filtered_hs
from evals.run_int import get_hs_proj

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
model_name = "gpt2-large"
concept = "number"
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
nsamples = 400

run_path = os.path.join(OUT, "run_output/number/gpt2-large/230627/run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")

run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl = load_model_eval(model_name, concept)

#l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
#test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
#    run["X_test"], run["facts_test"], run["foils_test"], 
#    l0_tl, l1_tl, nsamples=nsamples
#)
#l0_hs_wff = filter_hs_w_ys(
#    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
#)
#l1_hs_wff = filter_hs_w_ys(
#    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
#)
#if nsamples is not None:
#    l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)

#%%
l0_hs_wff, l1_hs_wff, other_hs = load_filtered_hs_wff(model_name, load_other=True)

#%%
all_concept_hs = [x for x,_,_ in l0_hs_wff + l1_hs_wff]
other_hs_no_x = [x for x,_ in other_hs]
all_hs = np.vstack(all_concept_hs + other_hs_no_x)

#%%
case = 0
nsamples=50
msamples=10

def compute_inner_loop(h, all_hs, P, I_P, V, msamples):
    all_pxnewh = []
    idx = np.random.choice(all_hs.shape[0], nsamples, replace=False)
    for other_h in all_hs[idx[:msamples]]:
        newh = other_h.T @ I_P + h.T @ P
        pxnewh = softmax(V @ newh)
        all_pxnewh.append(pxnewh)
    all_pxnewh = np.vstack(all_pxnewh).mean(axis=0)
    return all_pxnewh

def compute_concept_pxhs(c_hs, all_hs, I_P, P, V, msamples):
    c_pxhs = []
    for h,_,_ in tqdm(c_hs):
        inner_pxh = compute_inner_loop(h, all_hs, P, I_P, V, msamples)
        c_pxhs.append(inner_pxh)
    return np.vstack(c_pxhs)

l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
l0_pxhs = compute_concept_pxhs(l0_hs_wff, all_hs, I_P, P, V, msamples)
l1_pxhs = compute_concept_pxhs(l1_hs_wff, all_hs, I_P, P, V, msamples)


#%%
from data.create_p_x import load_p_x
from evals.kl_eval import renormalize
nucleus = False
p_x = load_p_x(model_name, nucleus)
l0_p_x = renormalize(p_x[l0_tl])
l1_p_x = renormalize(p_x[l1_tl])
h_l0_p_x = entropy(l0_p_x)
h_l1_p_x = entropy(l1_p_x)

#%%

def compute_entropies_pxhs(pxhs, case, l0_tl, l1_tl):
    hs_pxhs = []
    for q in pxhs:
    #q = pxhs[0]
        if case == 0:
            q_c = renormalize(q[l0_tl])
        elif case == 1:
            q_c = renormalize(q[l1_tl])
        else:
            raise ValueError("Incorrect case")
        hs_pxhs.append(entropy(q_c))
    return hs_pxhs

l0_hs_pxhs = compute_entropies_pxhs(l0_pxhs, 0, l0_tl, l1_tl)
l1_hs_pxhs = compute_entropies_pxhs(l1_pxhs, 1, l0_tl, l1_tl)
hs_pxhs = l0_hs_pxhs + l1_hs_pxhs

#%%
#nucleus = False
#h_x = entropy(p_x)
h_l0_p_x - l0_h_pxhs
h_l1_p_x - l1_h_pxhs

#%%
from evals.kl_eval import renormalize


#%%

from evals.kl_eval import get_all_distribs, get_distrib_key, renormalize
c_index = 0
h,_,_ = l0_hs_wff[0]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, V, l0_tl, l1_tl
)

key = get_distrib_key(c_index)
pxPh = renormalize(P_distribs[key])

# %%
