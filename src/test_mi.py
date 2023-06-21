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
#import pandas as pd
import pickle
#import torch
import random 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
from models.fit_kde import load_data
from data.process_generations_x import load_concept_token_lists
from scipy.special import softmax
from scipy.stats import entropy

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Computing correct MI #
####################
def get_p_c(model_name, I_P):
    cs_counts_path = os.path.join(OUT, f"p_x/{model_name}/c_counts_{model_name}_{I_P}.pkl")
    with open(cs_counts_path, "rb") as f:
        cs = pickle.load(f)

    counts = [cs["l0"], cs["l1"], cs["other"]]
    p = counts / np.sum(counts)
    return p 

def get_p_c_h(h, V, l0_tl, l1_tl):
    logits = V @ h
    probs = softmax(logits)

    p_0 = probs[l0_tl].sum()
    p_1 = probs[l1_tl].sum()
    p_other = np.delete(probs, np.hstack((l0_tl, l1_tl))).sum()
    p_c_h = np.array([p_0, p_1, p_other])
    return p_c_h

# %%

from evals.kl_eval import renormalize
#if __name__ == '__main__':
model_name = "gpt2-large"
#I_P = "no_I_P"
#nsamples = 10000

outdir = os.path.join(OUT, "new_mi_test")

#I_P = "I_P"
#I_P = "no_I_P"

V = get_V(model_name)
concept = get_concept_name(model_name)
l0_tl, l1_tl = load_concept_token_lists(concept, model_name)

from evals.kl_eval import load_run_Ps
from analysis.format_res import get_best_runs
P, I_P = load_run_Ps(get_best_runs(model_name, concept))

#%% MI COMPUTATION
def load_filtered_hs(model_name, I_P):
    filtered_hs_dir = os.path.join(OUT, 
        f"filtered_generations/{model_name}/{I_P}")
    with open(os.path.join(filtered_hs_dir, "l0_hs.pkl"), "rb") as f:
        l0_hs = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "l1_hs.pkl"), "rb") as f:
        l1_hs = pickle.load(f)
    return np.vstack(l0_hs), np.vstack(l1_hs)

nsamples=100
l0_hs, l1_hs = load_filtered_hs(model_name, "no_I_P")
np.random.shuffle(l0_hs)
np.random.shuffle(l1_hs)
ratio = l1_hs.shape[0]/l0_hs.shape[0]
l0_hs = l0_hs[:nsamples,:]
l1_hs = l1_hs[:int((nsamples*ratio)),:]

#def compute_pch(h, V, l0_tl, l1_tl):
#    p_c_h = get_p_c_h(h, V, l0_tl, l1_tl)
#    return 

def compute_all_pch(hs, c_index, P, V, l0_tl, l1_tl):
    base_pch_vals = []
    proj_pch_vals = []
    for i in tqdm(range(hs.shape[0])):
        #i = 0
        h = hs[i]
        
        base_pch = get_p_c_h(h, V, l0_tl, l1_tl)
        bin_base_pch = renormalize(base_pch[:2])
        base_pch_vals.append(bin_base_pch[c_index])

        proj_pch = get_p_c_h(np.matmul(P, h), V, l0_tl, l1_tl)
        bin_proj_pch = renormalize(proj_pch[:2])
        proj_pch_vals.append(bin_proj_pch[c_index])
    return base_pch_vals, proj_pch_vals

#H(C)
def get_h_c(model_name, I_P):
    p_c = renormalize(get_p_c(model_name, I_P)[:2])
    h_c = entropy(p_c)
    return p_c, h_c

p_c, h_c = get_h_c(model_name, "no_I_P")
l0_base_pch, l0_I_P_pch = compute_all_pch(l0_hs, 0, I_P, V, l0_tl, l1_tl)
l1_base_pch, l1_I_P_pch = compute_all_pch(l1_hs, 1, I_P, V, l0_tl, l1_tl)
base_pch = l0_base_pch + l1_base_pch
I_P_pch = l0_I_P_pch + l1_I_P_pch
base_h_c_h = -1 * np.log(np.array(base_pch)).mean()
I_P_h_c_h = -1 * np.log(np.array(I_P_pch)).mean()

base_mi = h_c - base_h_c_h
I_P_mi = h_c - I_P_h_c_h

#%% Accuracy w foil replacing fact
import random

def load_filtered_hs_wff(model_name, I_P):
    filtered_hs_dir = os.path.join(OUT, 
        f"filtered_generations/{model_name}/{I_P}")
    with open(os.path.join(filtered_hs_dir, "l0_hs_w_factfoil.pkl"), "rb") as f:
        l0_hs_wff = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "l1_hs_w_factfoil.pkl"), "rb") as f:
        l1_hs_wff = pickle.load(f)
    return l0_hs_wff, l1_hs_wff

def correct_flag(fact_prob, foil_prob):
    return (fact_prob > foil_prob)*1

def highest_rank(probs, id):
    probssortind = probs.argsort()
    return (probssortind[-1] == id)*1

def highest_verbs(probs, id, l0_tl, l1_tl):
    lemma_tl = np.hstack((l0_tl,l1_tl))
    lemma_probs = probs[lemma_tl]
    lemma_probs_sortind = lemma_probs.argsort()
    lemma_tl_sorted = lemma_tl[lemma_probs_sortind]
    return (lemma_tl_sorted[-1] == id) * 1

def compute_factfoil_flags(probs, proj_probs, fact_id, foil_id, l0_tl, l1_tl):
    return dict(
        orig_correct=correct_flag(probs[fact_id], probs[foil_id]),
        proj_correct=correct_flag(proj_probs[fact_id], proj_probs[foil_id]),
        orig_fact_highest=highest_rank(probs, fact_id),
        orig_foil_highest=highest_rank(probs, foil_id),
        proj_fact_highest=highest_rank(proj_probs, fact_id),
        proj_foil_highest=highest_rank(proj_probs, foil_id),
        orig_fact_highest_verbs=highest_verbs(probs, fact_id, l0_tl, l1_tl),
        orig_foil_highest_verbs=highest_verbs(probs, foil_id, l0_tl, l1_tl),
        proj_fact_highest_verbs=highest_verbs(proj_probs, fact_id, l0_tl, l1_tl),
        proj_foil_highest_verbs=highest_verbs(proj_probs, foil_id, l0_tl, l1_tl),
    )

def compute_various_accs(h, fact_id, foil_id, V, l0_tl, l1_tl):
    logits = V @ h.numpy()
    probs = softmax(logits)

    proj_logits = V @ np.matmul(I_P, h.numpy())
    proj_probs = softmax(proj_logits)

    flags = compute_factfoil_flags(
        probs, proj_probs, fact_id, foil_id, l0_tl, l1_tl
    )
    return flags

#%%
nsamples=100
l0_hs_wff, l1_hs_wff = load_filtered_hs_wff(model_name, "no_I_P")
random.shuffle(l0_hs_wff)
random.shuffle(l1_hs_wff)
ratio = len(l1_hs_wff)/len(l0_hs_wff)
l0_hs_wff = l0_hs_wff[:nsamples]
l1_hs_wff = l1_hs_wff[:int((nsamples*ratio))]

accs = []
all_hs = l0_hs_wff + l1_hs_wff
for h, faid, foid in tqdm(all_hs):
    accs.append(compute_various_accs(h, faid, foid, V, l0_tl, l1_tl))

import pandas as pd
accdf = pd.DataFrame(accs)
accdf.mean().to_csv(os.path.join(outdir, "avg_acc.csv"))
accdf.groupby(["orig_correct"]).mean().to_csv(os.path.join(outdir, "avg_acc_split.csv"))



#%% Loading files
"""
#%%
def sample_generated_hs(model_name, I_P, nsamples):
    X, weights = load_data(model_name, I_P)
    probs = weights / np.sum(weights)
    if nsamples is not None:
        idx = np.arange(0, X.shape[0])
        sampled_idx = np.random.choice(idx, size=nsamples, p=probs)
        X_samples = X[sampled_idx]
        X_p_h = probs[sampled_idx]
        return X_samples, X_p_h
    else:
        return X, probs

def compute_lm_mi(model_name, I_P, nsamples=None):
    logging.info(f"Sampling hs from generated hs.")
    X_samples, X_p_h = sample_generated_hs(model_name, I_P, nsamples)
    p_c = get_p_c(model_name, I_P)
    V = get_V(model_name)
    concept = get_concept_name(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept, model_name)

    all_mis = []
    logging.info(f"Computing MI")
    for i in tqdm(range(X_samples.shape[0])):
        h = X_samples[i]
        p_h = X_p_h[i]
        p_c_h = get_p_c_h(h, V, l0_tl, l1_tl)
        inner = p_c_h * np.log(np.divide(p_c_h, p_c))
        mi_val = p_h * inner.sum()
        all_mis.append(mi_val)
    return all_mis

model_name = "gpt2-large"
outdir = os.path.join(OUT, "new_mi")
I_P_mis_outfile = os.path.join(outdir, f"new_mis_{model_name}_I_P.npy")
I_P_mis = np.load(I_P_mis_outfile)

no_I_P_mis_outfile = os.path.join(outdir, f"new_mis_{model_name}_no_I_P.npy")
no_I_P_mis = np.load(no_I_P_mis_outfile)

# %%
p_c_no_I_P = get_p_c(model_name, "no_I_P")
p_c_I_P = get_p_c(model_name, "I_P")

# %%
"""