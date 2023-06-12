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
from scipy.stats import entropy

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from utils.dataset_loaders import load_hs, load_other_hs, load_model_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Loading          #
####################
def load_run_output(run_path):
    with open(run_path, 'rb') as f:      
        run = pickle.load(f)

    P = run["output"]["P_burn"]
    I_P = run["output"]["I_P_burn"]
    return P, I_P

#%%#################
# KL Helpers       #
####################
def get_logs(hidden_state, other_emb, l0_emb, l1_emb):
    other_log = other_emb @ hidden_state
    l0_log = l0_emb @ hidden_state
    l1_log = l1_emb @ hidden_state
    return other_log, l0_log, l1_log

def get_probs(hidden_state, other_emb, l0_emb, l1_emb):
    other_log, l0_log, l1_log = get_logs(
        hidden_state, other_emb, l0_emb, l1_emb
    )
    all_logits = np.hstack([other_log, l0_log, l1_log])
    all_probs = softmax(all_logits)
    other_probs = all_probs[:other_log.shape[0]]
    l0_end = other_log.shape[0] + l0_log.shape[0]
    l0_probs = all_probs[other_log.shape[0]:l0_end]
    l1_probs = all_probs[l0_end:]
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

def get_distribs(h, other_emb, l0_emb, l1_emb):
    all_split, other, l0, l1 = get_probs(h, other_emb, l0_emb, l1_emb)
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

def get_all_distribs(h, P, I_P, other_emb, l0_emb, l1_emb, X_pca=None):
    base_distribs = get_distribs(h, other_emb, l0_emb, l1_emb)
    if X_pca is None:
        P_distribs = get_distribs(P @ h, other_emb, l0_emb, l1_emb)
        I_P_distribs = get_distribs(I_P @ h, other_emb, l0_emb, l1_emb)
    else:
        h = h.reshape(1,-1)
        P_distribs = get_distribs(
            X_pca.inverse_transform(
                (P @ X_pca.transform(h).reshape(-1)).reshape(1, -1)).reshape(-1), 
            other_emb, l0_emb, l1_emb
        )
        I_P_distribs = get_distribs(
            X_pca.inverse_transform(
                (I_P @ X_pca.transform(h).reshape(-1)).reshape(1, -1)).reshape(-1), 
            other_emb, l0_emb, l1_emb
        )
    return base_distribs, P_distribs, I_P_distribs

def compute_kl(p, q, agg_func=np.sum):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        #return 0
        return agg_func(kl_div(p, q))
    else:
        return agg_func(kl_div(p, q))

def renormalize(p):
    return p / np.sum(p)

def compute_faith_kls(base, proj, prefix="", agg_func=np.sum):
    kls = {
        f"{prefix}faith_kl_all_split": compute_kl(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_kl_all_merged": compute_kl(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_kl_other": compute_kl(renormalize(base["other"]), renormalize(proj["other"]), agg_func),
        f"{prefix}faith_kl_tgt_split": compute_kl(renormalize(base["lemma_split"]), renormalize(proj["lemma_split"]), agg_func),
        f"{prefix}faith_kl_tgt_merged": compute_kl(renormalize(base["lemma_merged"]), renormalize(proj["lemma_merged"]), agg_func),
        f"{prefix}faith_kl_other_unnorm": compute_kl(base["other"], proj["other"], agg_func),
        f"{prefix}faith_kl_tgt_split_unnorm": compute_kl(base["lemma_split"], proj["lemma_split"], agg_func),
    }
    return kls
    
def compute_tvd(p, q, agg_func=np.sum):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        return 0
    else:
        return agg_func(np.abs(p-q))
    
def compute_faith_tvds(base, proj, prefix="", agg_func=np.sum):
    tvds = {
        f"{prefix}faith_tvd_all_split": compute_tvd(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_tvd_all_merged": compute_tvd(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_tvd_other": compute_tvd(renormalize(base["other"]), renormalize(proj["other"]), agg_func),
        f"{prefix}faith_tvd_tgt_split": compute_tvd(renormalize(base["lemma_split"]), renormalize(proj["lemma_split"]), agg_func),
        f"{prefix}faith_tvd_tgt_merged": compute_tvd(renormalize(base["lemma_merged"]), renormalize(proj["lemma_merged"]), agg_func),
    }
    return tvds

def compute_pct_chg(p, q, agg_func=np.mean):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        return 0
    else:
        return agg_func(np.abs(q-p) / p)
    
def compute_faith_pct_chg(base, proj, prefix="", agg_func=np.mean):
    pct_chgs = {
        f"{prefix}faith_pct_chg_all_split": compute_pct_chg(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_pct_chg_all_merged": compute_pct_chg(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_pct_chg_other": compute_pct_chg(renormalize(base["other"]), renormalize(proj["other"]), agg_func),
        f"{prefix}faith_pct_chg_tgt_split": compute_pct_chg(renormalize(base["lemma_split"]), renormalize(proj["lemma_split"]), agg_func),
        f"{prefix}faith_pct_chg_tgt_merged": compute_pct_chg(renormalize(base["lemma_merged"]), renormalize(proj["lemma_merged"]), agg_func),
    }
    return pct_chgs

# erasure KL
def compute_erasure_kl(base_pair_probs, proj_pair_probs, agg_func=np.sum):
    obs_er_kls = []
    for base_pair, proj_pair in zip(base_pair_probs, proj_pair_probs):
        obs_er_kls.append(compute_kl(base_pair, proj_pair, agg_func))
    return np.mean(obs_er_kls)

def compute_erasure_kls(base_pair_probs, proj_pair_probs, pair_probs, prefix="", agg_func=np.sum):
    erasure_kls = {
        f"{prefix}er_kl_base_proj": compute_erasure_kl(
            base_pair_probs, proj_pair_probs, agg_func),
        f"{prefix}er_kl_maj_base": compute_erasure_kl(
            pair_probs, base_pair_probs, agg_func),
        f"{prefix}er_kl_maj_proj": compute_erasure_kl(
            pair_probs, proj_pair_probs, agg_func)
    }
    return erasure_kls

# erasure pairwise MI
def compute_pairwise_entropy(pairwise_p):
    return np.apply_along_axis(entropy, 1, pairwise_p)

def compute_pairwise_mi(pairwise_uncond_ent, pairwise_cond_ent):
    return np.mean(pairwise_uncond_ent - pairwise_cond_ent)    

def get_pairwise_mi(pairwise_uncond_probs, pairwise_cond_probs):
    pairwise_uncond_ent = compute_pairwise_entropy(pairwise_uncond_probs)
    pairwise_cond_ent = compute_pairwise_entropy(pairwise_cond_probs)
    return compute_pairwise_mi(pairwise_uncond_ent, pairwise_cond_ent)

def get_all_pairwise_mis(verb_pairs_probs, base_pair_probs, P_pair_probs, I_P_pair_probs):
    res = dict(
        base_pairwise_mi = get_pairwise_mi(verb_pairs_probs, base_pair_probs),
        P_pairwise_mi = get_pairwise_mi(verb_pairs_probs, P_pair_probs),
        I_P_pairwise_mi = get_pairwise_mi(verb_pairs_probs, I_P_pair_probs)
    )
    return res

# erasure overall MI
def get_lemma_marginals(l0_prob, l1_prob):
    total_l0_prob = np.sum(l0_prob)
    total_l1_prob = np.sum(l1_prob)
    normalizer = total_l0_prob + total_l1_prob
    l0_l1_marginals = np.hstack([total_l0_prob, total_l1_prob]) / normalizer
    return l0_l1_marginals

def get_all_marginals(l0_prob, l1_prob, other_prob):
    total_l0_prob = np.sum(l0_prob)
    total_l1_prob = np.sum(l1_prob)
    total_other_prob = np.sum(other_prob)
    normalizer = total_l0_prob + total_l1_prob + total_other_prob
    all_marginals = np.hstack(
        [total_l0_prob, total_l1_prob, total_other_prob]) / normalizer
    return all_marginals

def compute_overall_mi(concept_marginals, cond_l0_probs, cond_l1_probs, cond_other_probs):
    all_marginals = [
        concept_marginals["p_0_incl_other"], 
        concept_marginals["p_1_incl_other"], 
        concept_marginals["p_other_incl_other"]
    ]
    cond_all_marginals = get_all_marginals(
        cond_l0_probs, cond_l1_probs, cond_other_probs
    )
    return entropy(all_marginals) - entropy(cond_all_marginals)

def compute_lemma_mi(concept_marginals, cond_l0_probs, cond_l1_probs):
    lemma_marginals = [
        concept_marginals["p_0_wout_other"], 
        concept_marginals["p_1_wout_other"]
    ]
    cond_lemma_marginals = get_lemma_marginals(
        cond_l0_probs, cond_l1_probs
    )
    return entropy(lemma_marginals) - entropy(cond_lemma_marginals)

def get_all_overall_mis(concept_marginals, base_distribs, P_distribs, I_P_distribs):
    res = dict(
        base_overall_mi = compute_overall_mi(
            concept_marginals, base_distribs["l0"], base_distribs["l1"], base_distribs["other"]),
        P_overall_mi = compute_overall_mi(
            concept_marginals, P_distribs["l0"], P_distribs["l1"],  base_distribs["other"]),
        I_P_overall_mi = compute_overall_mi(
            concept_marginals, I_P_distribs["l0"], I_P_distribs["l1"], base_distribs["other"]),
    )
    return res

def get_all_lemma_mis(concept_marginals, base_distribs, P_distribs, I_P_distribs):
    res = dict(
        base_lemma_mi = compute_lemma_mi(
            concept_marginals, base_distribs["l0"], base_distribs["l1"]),
        P_lemma_mi = compute_lemma_mi(
            concept_marginals, P_distribs["l0"], P_distribs["l1"]),
        I_P_lemma_mi = compute_lemma_mi(
            concept_marginals, I_P_distribs["l0"], I_P_distribs["l1"]),
    )
    return res

# main runners
def compute_all_faith_metrics(base_distribs, P_distribs, I_P_distribs):
    P_fth_kls = compute_faith_kls(base_distribs, P_distribs, prefix="P_")
    I_P_fth_kls = compute_faith_kls(base_distribs, I_P_distribs, prefix="I_P_")

    P_fth_tvds = compute_faith_tvds(base_distribs, P_distribs, prefix="P_")
    I_P_fth_tvds = compute_faith_tvds(base_distribs, I_P_distribs, prefix="I_P_")

    P_fth_pct_chg = compute_faith_pct_chg(base_distribs, P_distribs, prefix="P_")
    I_P_fth_pct_chg = compute_faith_pct_chg(base_distribs, I_P_distribs, prefix="I_P_")
    
    return P_fth_kls | P_fth_tvds | P_fth_pct_chg | I_P_fth_kls\
         | I_P_fth_tvds | I_P_fth_pct_chg

    
def compute_all_erasure_kls(base_distribs, P_distribs, I_P_distribs, pair_probs):
    base_pair_probs, P_pair_probs, I_P_pair_probs = get_all_pairwise_distribs(
        base_distribs, P_distribs, I_P_distribs
    )
    
    P_er_kls = compute_erasure_kls(
        base_pair_probs, P_pair_probs, pair_probs, prefix="P_"
    )
    I_P_er_kls = compute_erasure_kls(
        base_pair_probs, I_P_pair_probs, pair_probs, prefix="I_P_"
    )
    return P_er_kls | I_P_er_kls

def compute_all_erasure_mis(base_distribs, P_distribs, I_P_distribs, pair_probs, concept_marginals):
    base_pair_probs, P_pair_probs, I_P_pair_probs = get_all_pairwise_distribs(
        base_distribs, P_distribs, I_P_distribs
    )
    
    pairwise_mis = get_all_pairwise_mis(
        pair_probs, base_pair_probs, P_pair_probs, I_P_pair_probs
    )
    overall_mis = get_all_overall_mis(
        concept_marginals, base_distribs, P_distribs, I_P_distribs
    )
    lemma_mis = get_all_lemma_mis(
        concept_marginals, base_distribs, P_distribs, I_P_distribs
    )
    return pairwise_mis | overall_mis | lemma_mis

def compute_kls_one_sample(h, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, 
    concept_marginals, faith=True, er_kls=True, er_mis=True, X_pca=None):
    base_distribs, P_distribs, I_P_distribs = get_all_distribs(
        h, P, I_P, other_emb, l0_emb, l1_emb, X_pca
    )

    faith_metrics, er_kl_metrics, er_mis_metrics = {},{},{}
    if faith:
        faith_metrics = compute_all_faith_metrics(
            base_distribs, P_distribs, I_P_distribs
        )
    if er_kls:
        er_kl_metrics = compute_all_erasure_kls(
            base_distribs, P_distribs, I_P_distribs, pair_probs
        )
    if er_mis:
        er_mis_metrics = compute_all_erasure_mis(
            base_distribs, P_distribs, I_P_distribs, pair_probs, concept_marginals
        )
    return faith_metrics | er_kl_metrics | er_mis_metrics

def get_hs_sample_index(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    return idx[:nsamples]

def compute_kls(hs, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals, 
    nsamples=200, faith=True, er_kls=True, er_mis=True, X_pca = None):
    ind = get_hs_sample_index(hs, nsamples)    

    pbar = tqdm(ind)
    pbar.set_description("Computing faithfulness and erasure KL on hidden states")
    kls = []
    for i in pbar:
        kls.append(compute_kls_one_sample(
            hs[i], P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals,
            faith, er_kls, er_mis, X_pca
        ))
    kls = pd.DataFrame(kls)
    return kls


#%%#################
# Main             #
####################
if __name__ == '__main__':

    model_name = "bert-base-uncased"
    concept_name = "number"
    nsamples = 1000
    run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")

    logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

    concept_hs = load_hs(concept_name, model_name, nsamples=nsamples)
    other_hs = load_other_hs(concept_name, model_name, nsamples=nsamples)
    other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)
    P, I_P = load_run_output(run_output)
    
    concept_kls = compute_kls(concept_hs, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals)
    other_kls = compute_kls(other_hs, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals)
    
    concept_kls_desc = concept_kls.describe()
    concept_kls_desc.columns = ["concept_" + x for x in concept_kls_desc.columns]
    other_kls_desc = other_kls.describe()
    other_kls_desc.columns = ["other_" + x for x in other_kls_desc.columns]
    
    all_kls = pd.concat([concept_kls, other_kls], axis=0)
    all_kls_desc = all_kls.describe()
    all_kls_desc.columns = ["all_" + x for x in all_kls_desc.columns]
    
    kls = pd.concat([concept_kls_desc, other_kls_desc, all_kls_desc], axis=1)

    kls.to_csv(os.path.join(OUT, "run_kls.csv"))
