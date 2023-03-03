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

from scipy.special import softmax, kl_div

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# Loading          #
####################

def load_hs(dataset_name, model_name):
    if model_name == "gpt2":
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}_{model_name}_ar.pkl")
    elif model_name == "bert-base-uncased":
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}_{model_name}_masked.pkl")
    else:
        DATASET = None

    with open(DATASET, 'rb') as f:      
        data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])
        h = np.array([x for x in data["h"]])
        del data
    return h

def sample_hs(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    ind = idx[:nsamples]
    return hs[ind]

def load_model_eval(model_name):
    WORD_EMB = os.path.join(DATASETS, f"processed/linzen_word_lists/{model_name}_word_embeds.npy")
    VERB_P = os.path.join(DATASETS, f"processed/linzen_word_lists/{model_name}_verb_p.npy")
    SG_EMB = os.path.join(DATASETS, f"processed/linzen_word_lists/{model_name}_sg_embeds.npy")
    PL_EMB = os.path.join(DATASETS, f"processed/linzen_word_lists/{model_name}_pl_embeds.npy")

    word_emb = np.load(WORD_EMB)
    verb_p = np.load(VERB_P)
    sg_emb = np.load(SG_EMB)
    pl_emb = np.load(PL_EMB)
    return word_emb, sg_emb, pl_emb, verb_p

def load_run_output(run_path):
    with open(run_path, 'rb') as f:      
        run = pickle.load(f)

    P = run["diag_rlace_usage_eval"]["P"]
    I_P = run["diag_rlace_usage_eval"]["I_P"]
    return P, I_P

#%%#################
# KL Helpers       #
####################
def get_logs(hidden_state, word_emb, sg_emb, pl_emb):
    word_log = word_emb @ hidden_state
    sg_log = sg_emb @ hidden_state
    pl_log = pl_emb @ hidden_state
    return word_log, sg_log, pl_log

def get_probs(hidden_state, word_emb, sg_emb, pl_emb):
    word_log, sg_log, pl_log = get_logs(
        hidden_state, word_emb, sg_emb, pl_emb
    )
    all_logits = np.hstack([word_log, sg_log, pl_log])
    all_probs = softmax(all_logits)
    word_probs = all_probs[:word_log.shape[0]]
    sg_end = word_log.shape[0] + sg_log.shape[0]
    sg_probs = all_probs[word_log.shape[0]:sg_end]
    pl_probs = all_probs[sg_end:]
    return all_probs, word_probs, sg_probs, pl_probs

def get_merged_probs(word_probs, sg_probs, pl_probs):
    lemma_merged = sg_probs + pl_probs
    all_merged = np.hstack([word_probs,lemma_merged])
    return lemma_merged, all_merged

def normalize_pairs(sg, pl):
    base_pair = np.vstack((sg, pl)).T
    base_pair_Z = np.sum(base_pair,axis=1)
    base_pair_probs = np.vstack([
        np.divide(base_pair[:,0], base_pair_Z),
        np.divide(base_pair[:,1], base_pair_Z)]
    ).T
    return base_pair_probs

def get_all_distribs(hidden_state, word_emb, sg_emb, pl_emb):
    all_split, words, sg, pl = get_probs(hidden_state, word_emb, sg_emb, pl_emb)
    lemma_split = np.hstack([sg, pl])
    lemma_merged, all_merged = get_merged_probs(
        words, sg, pl
    )
    return dict(
        words=words,
        sg=sg,
        pl=pl,
        all_split=all_split,
        all_merged=all_merged,
        lemma_split=lemma_split,
        lemma_merged=lemma_merged
    )

def compute_faith_kls(base, proj, prefix=""):
    kls = {
        f"{prefix}faith_kl_all_split": np.mean(kl_div(base["all_split"], proj["all_split"])),
        f"{prefix}faith_kl_all_merged": np.mean(kl_div(base["all_merged"], proj["all_merged"])),
        f"{prefix}faith_kl_words": np.mean(kl_div(base["words"], proj["words"])),
        f"{prefix}faith_kl_tgt_split": np.mean(kl_div(base["lemma_split"], proj["lemma_split"])),
        f"{prefix}faith_kl_tgt_merged": np.mean(kl_div(base["lemma_merged"], proj["lemma_merged"])),
    }
    return kls
    
# erasure KL
def compute_erasure_kl(base_pair_probs, proj_pair_probs):
    obs_er_kls = []
    for base_pair, proj_pair in zip(base_pair_probs, proj_pair_probs):
        obs_er_kls.append(np.mean(kl_div(base_pair, proj_pair)))
    return np.mean(obs_er_kls)

def compute_erasure_kls(base_pair_probs, proj_pair_probs, verb_probs, prefix=""):
    erasure_kls = {
        f"{prefix}er_kl_base_proj": compute_erasure_kl(
            base_pair_probs, proj_pair_probs),
        f"{prefix}er_kl_maj_base": compute_erasure_kl(
            verb_probs, base_pair_probs),
        f"{prefix}er_kl_maj_proj": compute_erasure_kl(
            verb_probs, proj_pair_probs)
    }
    return erasure_kls

def compute_kls_one_sample(h, P, I_P, word_emb, sg_emb, pl_emb, verb_probs):
    base_distribs = get_all_distribs(h, word_emb, sg_emb, pl_emb)
    P_distribs = get_all_distribs(P @ h, word_emb, sg_emb, pl_emb)
    I_P_distribs = get_all_distribs(I_P @ h, word_emb, sg_emb, pl_emb)

    P_fth_kls = compute_faith_kls(base_distribs, P_distribs, prefix="P_")
    I_P_fth_kls = compute_faith_kls(base_distribs, I_P_distribs, prefix="I_P_")

    base_pair_probs = normalize_pairs(base_distribs["sg"], base_distribs["pl"])
    P_pair_probs = normalize_pairs(P_distribs["sg"], P_distribs["pl"])
    I_P_pair_probs = normalize_pairs(I_P_distribs["sg"], I_P_distribs["pl"])

    P_er_kls = compute_erasure_kls(
        base_pair_probs, P_pair_probs, verb_probs, prefix="P_"
    )
    I_P_er_kls = compute_erasure_kls(
        base_pair_probs, I_P_pair_probs, verb_probs, prefix="I_P_"
    )

    return P_fth_kls | P_er_kls | I_P_fth_kls | I_P_er_kls

def compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    ind = idx[:nsamples]

    pbar = tqdm(ind)
    pbar.set_description("Computing faithfulness and erasure KL on hidden states")
    kls = []
    for i in pbar:
        kls.append(compute_kls_one_sample(hs[i], P, I_P, word_emb, sg_emb, pl_emb, verb_probs))
    kls = pd.DataFrame(kls).describe()
    return kls

#%%#################
# Main             #
####################
if __name__ == '__main__':

    model_name = "gpt2"
    dataset_name = "linzen"
    run_output = os.path.join(OUT, "run_output/gpt2/230302/run_gpt2_k_1_plr_0.001_clflr_0.0003_bs_256_0_1.pkl")

    logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

    hs = load_hs(dataset_name, model_name)
    word_emb, sg_emb, pl_emb, verb_probs = load_model_eval(model_name)
    P, I_P = load_run_output(run_output)
    
    kls = compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs)
    kls.to_csv(os.path.join(OUT, "run_kls.csv"))
