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


#%% Step 1


#def process_hs(hs_wff, P, I_P = None):
#    hs = []
#    Phs = []
#    #I_Phs = []
#    for h, _, _ in hs_wff:
#        hs.append(h)
#        Phs.append(h.T @ P)
#        #I_Phs.append(h.T @ I_P)
#    hs = np.vstack(hs)
#    P_hs = np.vstack(Phs)
#    #I_P_hs = np.vstack(I_Phs)
#    return hs, P_hs#, I_P_hs

def compute_logits(V, hs, P_hs, I_P_hs=None):
    logits = (hs @ V.T)
    P_logits = (P_hs @ V.T)
    #I_P_logits = (I_P_hs @ V.T)
    return logits, P_logits#, I_P_logits

def compute_norm_variance(hs, ord=2):
    return np.var(np.linalg.norm(hs, ord=ord, axis=1))

def compute_hs_logits(hs_wff, P, V, I_P=None):
    #hs, P_hs, I_P_hs = process_hs(hs_wff, P, I_P)
    #logits, P_logits, I_P_logits = compute_logits(
    #    V, hs, P_hs, I_P_hs
    #)
    hs = [x[0] for x in hs_wff]
    P_hs = get_hs_proj(hs_wff, P)
    logits, P_logits = compute_logits(
        V, hs, P_hs
    )
    return hs, P_hs, logits, P_logits

def compute_random_hs_logits(P_shape, nsamples, l0_hs_wff, l1_hs_wff):
    l0_randP_h_var = []
    #randP_logits_var = []
    l0_randP_h_normvar = []
    #randP_logits_normvar = []
    l1_randP_h_var = []
    #randP_logits_var = []
    l1_randP_h_normvar = []
    #randP_logits_normvar = []
    for i in tqdm(range(nsamples)):
        v = np.random.normal(size=P_shape)
        nv = v / np.linalg.norm(v)
        randP = nv.reshape(-1,1) @ nv.reshape(1,-1)
        #print(proj.shape)

        _, l0_randP_hs, _, _ = compute_hs_logits(
            l0_hs_wff, randP, V
        )
        _, l1_randP_hs, _, _ = compute_hs_logits(
            l1_hs_wff, randP, V
        )

        l0_randP_h_var.append(np.var(l0_randP_hs))
        l1_randP_h_var.append(np.var(l1_randP_hs))
        l0_randP_h_normvar.append(compute_norm_variance(l0_randP_hs))
        l1_randP_h_normvar.append(compute_norm_variance(l1_randP_hs))

    l0_randP_h_var = np.mean(l0_randP_h_var)
    l1_randP_h_var = np.mean(l1_randP_h_var)
    l0_randP_h_normvar = np.mean(l0_randP_h_normvar)
    l1_randP_h_normvar = np.mean(l1_randP_h_normvar)
    return l0_randP_h_var, l1_randP_h_var, l0_randP_h_normvar, l1_randP_h_normvar


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
l0_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
)
l1_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
)
if nsamples is not None:
    l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)


l0_hs, l0_P_hs, _, _ = compute_hs_logits(l0_hs_wff, P, V)
l1_hs, l1_P_hs, _, _ = compute_hs_logits(l1_hs_wff, P, V)


l0_rand_var, l1_rand_var, l0_rand_normvar, l1_rand_normvar = compute_random_hs_logits(
    P.shape[0], 3, l0_hs_wff, l1_hs_wff
)

resdict = {
    "l0_var_hs": np.var(l0_hs),
    "l0_var_P_hs": np.var(l0_P_hs),
    "l0_var_randP_hs": l0_rand_var,
    "l0_l2var_hs": compute_norm_variance(l0_hs),
    "l0_l2var_P_hs": compute_norm_variance(l0_P_hs),
    "l0_l2var_randP_hs": l0_rand_normvar,
    "l1_var_hs": np.var(l1_hs),
    "l1_var_P_hs": np.var(l1_P_hs),
    "l1_var_randP_hs": l1_rand_var,
    "l1_l2var_hs": compute_norm_variance(l1_hs),
    "l1_l2var_P_hs": compute_norm_variance(l1_P_hs),
    "l1_l2var_randP_hs": l1_rand_normvar
}

#%%
print("Singular hs:")
print(f"Variance of hs: {np.var(l0_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l0_P_hs):9.4f}")
print(f"Variance of random I-P hs: {l0_rand_var:9.4f}")
#print(f"Variance of logits: {np.var(l0_logits):9.4f}")
#print(f"Variance of I-P logits: {np.var(l0_P_logits):9.4f}")
#print(f"Variance of random I-P logits: {np.var(l0_randP_logits):9.4f}")

print(f"2-norm Variance of hs: {compute_norm_variance(l0_hs):9.4f}")
print(f"2-norm Variance of I-P hs: {compute_norm_variance(l0_P_hs):9.4f}")
print(f"2-norm Variance of random I-P hs: {l0_rand_normvar:9.4f}")
#print(f"2-norm Variance of logits: {compute_norm_variance(l0_logits):9.4f}")
#print(f"2-norm Variance of I-P logits: {compute_norm_variance(l0_P_logits):9.4f}")
#print(f"2-norm Variance of random I-P logits: {compute_norm_variance(l0_randP_logits):9.4f}")


print("Plural hs:")
print(f"Variance of hs: {np.var(l1_hs):9.4f}")
print(f"Variance of I-P hs: {np.var(l1_P_hs):9.4f}")
print(f"Variance of random I-P hs: {l1_rand_var:9.4f}")
#print(f"Variance of logits: {np.var(l1_logits):9.4f}")
#print(f"Variance of I-P logits: {np.var(l1_P_logits):9.4f}")
#print(f"Variance of random I-P logits: {np.var(l1_randP_logits):9.4f}")


print(f"2-norm Variance of hs: {compute_norm_variance(l1_hs):9.4f}")
print(f"2-norm Variance of I-P hs: {compute_norm_variance(l1_P_hs):9.4f}")
print(f"2-norm Variance of random I-P hs: {l1_rand_normvar:9.4f}")
#print(f"2-norm Variance of logits: {compute_norm_variance(l1_logits):9.4f}")
#print(f"2-norm Variance of I-P logits: {compute_norm_variance(l1_P_logits):9.4f}")
#print(f"2-norm Variance of random I-P logits: {compute_norm_variance(l1_randP_logits):9.4f}")


#%%
import seaborn as sns

sns.kdeplot(l0_logits.flatten()[:1000000])
sns.kdeplot(l1_logits.flatten()[:1000000])

sns.kdeplot(l0_P_logits.flatten()[:1000000])
sns.kdeplot(l1_P_logits.flatten()[:1000000])

sns.kdeplot(l0_randP_logits.flatten()[:1000000])
sns.kdeplot(l1_randP_logits.flatten()[:1000000])