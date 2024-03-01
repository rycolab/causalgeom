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
import torch

from scipy.special import softmax, kl_div
from scipy.stats import entropy
from transformers import TopPLogitsWarper, LogitsProcessorList

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT
from data.embed_wordlists.embedder_paired import load_concept_token_lists
from utils.lm_loaders import get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Loading          #
####################
def load_run_output(run_path):
    with open(run_path, 'rb') as f:      
        run = pickle.load(f)
    return run

def load_run_Ps(run_path):
    run = load_run_output(run_path)
    P = run["output"]["P"]
    I_P = run["output"]["I_P"]
    bias = run["output"].get("bias", None)
    return P, I_P, bias

def load_model_eval(model_name, concept, single_token):
    V = get_V(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept, model_name, single_token)
    return V, l0_tl, l1_tl


#%%#################
# Distrib Helpers  #
####################
def get_probs(hidden_state, V, l0_tl, l1_tl, processor=None):
    logits = V @ hidden_state
    if processor is not None:
        logits = torch.FloatTensor(logits).unsqueeze(0)
        tokens = torch.LongTensor([0]).unsqueeze(0)
        logits = processor(tokens, logits).squeeze(0).numpy()
    all_probs = softmax(logits)
    l0_probs = all_probs[l0_tl]
    l1_probs = all_probs[l1_tl]
    other_probs = np.delete(all_probs, np.hstack((l0_tl, l1_tl)))
    return all_probs, other_probs, l0_probs, l1_probs

def get_merged_probs(other_probs, l0_probs, l1_probs):
    lemma_merged = l0_probs + l1_probs
    all_merged = np.hstack([other_probs,lemma_merged])
    return lemma_merged, all_merged

def normalize_pairs(l0, l1):
    base_pair = np.vstack((l0, l1)).T
    base_pair_Z = np.sum(base_pair,axis=1)
    base_pair_probs = np.vstack([
        np.divide(base_pair[:,0], base_pair_Z),
        np.divide(base_pair[:,1], base_pair_Z)]
    ).T
    return base_pair_probs

def get_all_pairwise_distribs(base_distribs, P_distribs, I_P_distribs):
    base_pair_probs = normalize_pairs(base_distribs["l0"], base_distribs["l1"])
    P_pair_probs = normalize_pairs(P_distribs["l0"], P_distribs["l1"])
    I_P_pair_probs = normalize_pairs(I_P_distribs["l0"], I_P_distribs["l1"])
    return base_pair_probs, P_pair_probs, I_P_pair_probs

def get_distribs(h, V, l0_tl, l1_tl, processor=None):
    all_split, other, l0, l1 = get_probs(h, V, l0_tl, l1_tl, processor)
    lemma_split = np.hstack([l0, l1])
    lemma_merged, all_merged = get_merged_probs(
        other, l0, l1
    )
    return dict(
        other=other,
        l0=l0,
        l1=l1,
        all_split=all_split,
        all_merged=all_merged,
        lemma_split=lemma_split,
        lemma_merged=lemma_merged
    )

def get_all_distribs(h, P, I_P, V, l0_tl, l1_tl, processor=None):
    base_distribs = get_distribs(h, V, l0_tl, l1_tl, processor)
    P_distribs = get_distribs(h.T @ P, V, l0_tl, l1_tl, processor)
    I_P_distribs = get_distribs(h.T @ I_P, V, l0_tl, l1_tl, processor)
    return base_distribs, P_distribs, I_P_distribs

def renormalize(p):
    return p / np.sum(p)


#%%###########
# MI helpers #
##############


#%%######################
# Accuracy computations #
#########################
def correct_flag(fact_prob, foil_prob):
    return (fact_prob > foil_prob)*1

def highest_rank(probs, id):
    probssortind = probs.argsort()
    return (probssortind[-1] == id)*1

def highest_concept(probs, id, l0_tl, l1_tl):
    lemma_tl = np.hstack((l0_tl,l1_tl))
    lemma_probs = probs[lemma_tl]
    lemma_probs_sortind = lemma_probs.argsort()
    lemma_tl_sorted = lemma_tl[lemma_probs_sortind]
    return (lemma_tl_sorted[-1] == id) * 1