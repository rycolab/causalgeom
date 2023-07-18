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
from data.filter_generations import load_filtered_hs_wff

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
model_name = "gpt2-large"
concept = "number"
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
nsamples = 400

#%%
from evals.eval_run import filter_hs_w_ys, sample_filtered_hs

run_path = os.path.join(OUT, "run_output/number/gpt2-large/230627/run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")

run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl = load_model_eval(model_name, concept)

#%%
from utils.lm_loaders import get_tokenizer

tokenizer = get_tokenizer(model_name)
l0_decode = tokenizer.decode(l0_tl)
l1_decode = tokenizer.decode(l1_tl)
for l0,l1 in zip(l0_decode, l1_decode):
    print(l0, l1)

#l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
#test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
#    run["X_test"], run["facts_test"], run["foils_test"], 
#    l0_tl, l1_tl, nsamples=nsamples
#)
l0_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
)
l1_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
)
if nsamples is not None:
    l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)

#%% Step 1
def process_hs(hs_wff, P, I_P):
    hs = []
    Phs = []
    I_Phs = []
    for h, _, _ in hs_wff:
        hs.append(h)
        Phs.append(h.T @ P)
        I_Phs.append(h.T @ I_P)
    hs = np.vstack(hs)
    P_hs = np.vstack(Phs)
    I_P_hs = np.vstack(I_Phs)
    return hs, P_hs, I_P_hs

def compute_logits(V, hs, P_hs, I_P_hs):
    logits = (hs @ V.T)
    P_logits = (P_hs @ V.T)
    I_P_logits = (I_P_hs @ V.T)
    return logits, P_logits, I_P_logits

l0_hs, l0_P_hs, l0_I_P_hs = process_hs(l0_hs_wff, P, I_P)
l0_logits, l0_P_logits, l0_I_P_logits = compute_logits(
    V, l0_hs, l0_P_hs, l0_I_P_hs
)
l1_hs, l1_P_hs, l1_I_P_hs = process_hs(l1_hs_wff, P, I_P)
l1_logits, l1_P_logits, l1_I_P_logits = compute_logits(
    V, l1_hs, l1_P_hs, l1_I_P_hs
)
#%%
print("Singular hs:")
print(f"Variance of hs: {np.var(l0_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l0_P_hs):9.4f}")
print(f"Variance of P hs: {np.var(l0_I_P_hs):9.4f}")
print(f"Variance of logits: {np.var(l0_logits):9.4f}")
print(f"Variance of I-P logits: {np.var(l0_P_logits):9.4f}")
print(f"Variance of P hs: {np.var(l0_I_P_logits):9.4f}")

print("Plural hs:")
print(f"Variance of hs: {np.var(l1_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l1_P_hs):9.4f}")
print(f"Variance of P hs: {np.var(l1_I_P_hs):9.4f}")
print(f"Variance of logits: {np.var(l1_logits):9.4f}")
print(f"Variance of I-P logits: {np.var(l1_P_logits):9.4f}")
print(f"Variance of P hs: {np.var(l1_I_P_logits):9.4f}")


#%%
import seaborn as sns

sns.kdeplot(l0_logits.flatten()[:1000000])
sns.kdeplot(l1_logits.flatten()[:1000000])

sns.kdeplot(l0_P_logits.flatten()[:1000000])
sns.kdeplot(l1_P_logits.flatten()[:1000000])

# %%
randP_h_var = []
randP_logits_var = []
for i in range(100):
    v = np.random.normal(size=P.shape[0])
    nv = v / np.linalg.norm(v)
    proj = nv.reshape(-1,1) @ nv.reshape(1,-1)
    print(proj.shape)

    l0_hs, l0_randP_hs, l0_eye_hs = process_hs(l0_hs_wff, proj, np.eye(P.shape[0]))
    l0_logits, l0_randP_logits, l0_eye_logits = compute_logits(
        V, l0_hs, l0_randP_hs, l0_eye_hs
    )

    l1_hs, l1_randP_hs, l1_eye_hs = process_hs(l1_hs_wff, proj, np.eye(P.shape[0]))
    l1_logits, l1_randP_logits, l1_eye_logits = compute_logits(
        V, l1_hs, l1_randP_hs, l1_eye_hs
    )
    randP_h_var.append(np.var(l0_randP_hs))
    randP_logits_var.append(np.var(l1_randP_logits))

#%%
print("Singular hs:")
print(f"Variance of hs: {np.var(l0_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l0_P_hs):9.4f}")
print(f"Variance of random I-P hs: {np.var(l0_randP_hs):9.4f}")
print(f"Variance of logits: {np.var(l0_logits):9.4f}")
print(f"Variance of I-P logits: {np.var(l0_P_logits):9.4f}")
print(f"Variance of random I-P logits: {np.var(l0_randP_logits):9.4f}")


print("Plural hs:")
print(f"Variance of hs: {np.var(l1_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l1_P_hs):9.4f}")
print(f"Variance of random I-P hs: {np.var(l1_randP_hs):9.4f}")
print(f"Variance of logits: {np.var(l1_logits):9.4f}")
print(f"Variance of I-P logits: {np.var(l1_P_logits):9.4f}")
print(f"Variance of random I-P logits: {np.var(l1_randP_logits):9.4f}")

# %%
sns.kdeplot(l0_randP_logits.flatten()[:1000000])
sns.kdeplot(l1_randP_logits.flatten()[:1000000])