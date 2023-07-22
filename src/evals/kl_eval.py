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
from utils.lm_loaders import get_tokenizer, get_V, BERT_LIST, GPT2_LIST
from utils.dataset_loaders import load_hs, load_other_hs
from data.embed_wordlists.embedder import load_concept_token_lists
#from evals.eval_loaders import load_model_eval
from data.filter_generations import load_filtered_hs_wff

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
    P = run["output"]["P_burn"]
    I_P = run["output"]["I_P_burn"]
    return P, I_P

"""
def get_p_c_path_ancestral(concept, model_name):
    if model_name == "gpt2-large" and concept == "number":
        return os.path.join(OUT, f"p_x/{model_name}/c_counts_{model_name}_no_I_P.pkl")
    elif model_name == "gpt2-base-french" and concept == "gender":
        return os.path.join(OUT, f"p_x/{model_name}/c_counts_{model_name}_no_I_P.pkl")
    else:
        raise ValueError(f"No ancestral sampling concept marginal for {model_name} and {concept_name}")

def get_p_c_path_nucleus(concept, model_name):
    if model_name == "gpt2-large" and concept == "number":
        #return os.path.join(OUT, f"p_x/{model_name}/c_counts_{model_name}_no_I_P.pkl")
        return None
    elif model_name == "gpt2-base-french" and concept == "gender":
        #return os.path.join(OUT, f"p_x/{model_name}/c_counts_{model_name}_no_I_P.pkl")
        return None
    else:
        raise ValueError(f"No nucleus sampling concept marginal for {model_name} and {concept_name}")

def get_p_c_path_curated(concept, model_name):
    if model_name in ["bert-base-uncased", "gpt2-large"] and concept == "number":
        return os.path.join(DATASETS, "processed/en/word_lists/number_marginals.pkl")
    elif model_name in ["camembert-base", "gpt2-base-french"] and concept == "gender":
        return os.path.join(DATASETS, "processed/fr/word_lists/gender_marginals.pkl")
    else:
        raise ValueError(f"No curated dataset concept marginal for {model_name} and {concept_name}")

def get_p_c_path(concept, model_name, source):
    if source == "ancestral":
        return get_p_c_path_ancestral(concept, model_name)
    elif source == "nucleus":
        return get_p_c_path_nucleus(concept, model_name)
    elif source == "curated":
        return get_p_c_path_curated(concept, model_name)
    else:
        raise ValueError(f"Invalid source argument for get_p_c")

def get_p_c(concept, model_name, source):
    p_c_path = get_p_c_path(concept, model_name, source)
    if source == "curated":
        with open(p_c_path, 'rb') as f:
            concept_marginals = pickle.load(f)
        p = np.array([
            concept_marginals["p_0_incl_other"], 
            concept_marginals["p_1_incl_other"], 
            concept_marginals["p_other_incl_other"]
        ])
    else:
        with open(p_c_path, "rb") as f:
            cs = pickle.load(f)
        counts = [cs["l0"], cs["l1"], cs["other"]]
        p = counts / np.sum(counts)
    return p 
"""
def load_model_eval(model_name, concept):
    V = get_V(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
    #p_c = get_p_c(concept, model_name, source)
    return V, l0_tl, l1_tl#, p_c

#%%#################
# Distrib Helpers  #
####################
def get_probs(hidden_state, V, l0_tl, l1_tl):
    logits = V @ hidden_state
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

def get_distribs(h, V, l0_tl, l1_tl):
    all_split, other, l0, l1 = get_probs(h, V, l0_tl, l1_tl)
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

def get_all_distribs(h, P, I_P, V, l0_tl, l1_tl):
    base_distribs = get_distribs(h, V, l0_tl, l1_tl)
    P_distribs = get_distribs(h.T @ P, V, l0_tl, l1_tl)
    I_P_distribs = get_distribs(h.T @ I_P, V, l0_tl, l1_tl)
    return base_distribs, P_distribs, I_P_distribs

def renormalize(p):
    return p / np.sum(p)


#%%#################
# KL Helpers       #
####################
def compute_kl(p, q, agg_func=np.sum):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        #return 0
        return agg_func(kl_div(p, q))
    else:
        return agg_func(kl_div(p, q))

def compute_faith_kls(base, proj, prefix="", agg_func=np.sum):
    kls = {
        f"{prefix}faith_kl_all_split": compute_kl(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_kl_all_merged": compute_kl(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_kl_other": compute_kl(renormalize(base["other"]), renormalize(proj["other"]), agg_func),
        f"{prefix}faith_kl_tgt_split": compute_kl(renormalize(base["lemma_split"]), renormalize(proj["lemma_split"]), agg_func),
        f"{prefix}faith_kl_tgt_merged": compute_kl(renormalize(base["lemma_merged"]), renormalize(proj["lemma_merged"]), agg_func),
        f"{prefix}faith_kl_other_unnorm": compute_kl(base["other"], proj["other"], agg_func),
        f"{prefix}faith_kl_tgt_split_unnorm": compute_kl(base["lemma_split"], proj["lemma_split"], agg_func),
        f"{prefix}faith_kl_tgt_merged_unnorm": compute_kl(base["lemma_merged"], proj["lemma_merged"], agg_func),
    }
    return kls
    
def compute_tvd(p, q, agg_func=np.sum):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before TVD")
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
        logging.warn("Distribution not normalized before PCTCHG")
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
"""
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
"""
#%% Erasure MI helpers
def get_bin_p_c_h(l0_prob, l1_prob):
    """ get p(c|h) for binary c"""
    total_l0_prob = np.sum(l0_prob)
    total_l1_prob = np.sum(l1_prob)
    return renormalize(np.hstack([total_l0_prob, total_l1_prob]))

#def compute_surprisal(bin_p_c_h, c_index):
#    return -1 * np.log(bin_p_c_h[c_index])

#def compute_p_c_h_surprisal(l0_prob, l1_prob, c_index):
#    bin_p_c_h = get_bin_p_c_h(l0_prob, l1_prob)
#    return compute_surprisal(bin_p_c_h, c_index)

#def compute_all_p_c_h_surprisals(base_distribs, P_distribs, I_P_distribs, c_index):
#    return dict(
#        base_h_c_h = compute_p_c_h_surprisal(
#            base_distribs["l0"], base_distribs["l1"], c_index),
#        P_h_c_h = compute_p_c_h_surprisal(
#            P_distribs["l0"], P_distribs["l1"], c_index),
#        I_P_h_c_h = compute_p_c_h_surprisal(
#            I_P_distribs["l0"], I_P_distribs["l1"], c_index),
#    )

def compute_h_c_h(l0_prob, l1_prob):
    bin_p_c_h = get_bin_p_c_h(l0_prob, l1_prob)
    return entropy(bin_p_c_h)

def compute_all_h_c_h(base_distribs, P_distribs, I_P_distribs):
    return dict(
        base_h_c_h = compute_h_c_h(
            base_distribs["l0"], base_distribs["l1"]),
        P_h_c_h = compute_h_c_h(
            P_distribs["l0"], P_distribs["l1"]),
        I_P_h_c_h = compute_h_c_h(
            I_P_distribs["l0"], I_P_distribs["l1"]),
    )

def get_h_c_bin(p_c):
    p_c = renormalize(p_c[:2])
    h_c = entropy(p_c)
    return h_c

def compute_h_c_bin(l0_hs, l1_hs):
    c_counts = np.array([len(l0_hs), len(l1_hs)])
    p_c = c_counts / np.sum(c_counts)
    h_c = get_h_c_bin(p_c)
    return h_c

def compute_mi(h_c, full_eval_means):
    full_eval_means["h_c"] = h_c
    for mi_type in ["base", "P", "I_P"]:
        full_eval_means[f"{mi_type}_mi"] = h_c - full_eval_means[f"{mi_type}_h_c_h"]

#def compute_mi_x(h_x, full_eval):
#    full_eval["h_x"] = h_x
#    for mi_type in ["P", "I_P"]:
#        full_eval[f"mi_x_{mi_type}h_c"] = h_x - full_eval[f"h_p_x_{mi_type}h"]

def compute_h_x(p_x, l0_tl, l1_tl, full_eval_means):
    l0_p_x = renormalize(p_x[l0_tl])
    l1_p_x = renormalize(p_x[l1_tl])
    h_l0_p_x = entropy(l0_p_x)
    h_l1_p_x = entropy(l1_p_x)
    full_eval_means["h_p_x_l0"] = h_l0_p_x
    full_eval_means["h_p_x_l1"] = h_l1_p_x

#%% FTH MI Helpers
def get_distrib_key(c_index):
    if c_index == 0:
        key = "l0"
    elif c_index == 1:
        key = "l1"
    else:
        raise ValueError(f"Wrong c_index value {c_index}, has to be {0, 1}")
    return key

def compute_faith_mi(base_distribs, P_distribs, I_P_distribs, c_index):
    key = get_distrib_key(c_index)
    pxh = renormalize(base_distribs[key])
    pxPh = renormalize(P_distribs[key])
    pxI_Ph = renormalize(I_P_distribs[key])
    log_pxh = np.log(pxh)
    log_pxPh = np.log(pxPh)
    log_pxI_Ph = np.log(pxI_Ph)

    return dict(
        h_p_x_Ph = entropy(pxPh),
        h_p_x_I_Ph = entropy(pxI_Ph),
        P_fth_mi = np.sum(pxh * (log_pxh - log_pxPh)),
        I_P_fth_mi = np.sum(pxh * (log_pxh - log_pxI_Ph))
    )

#%% Accuracy computations
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

def compute_factfoil_flags(probs, fact_id, foil_id, l0_tl, l1_tl, prefix):
    return {
        f"{prefix}_acc_correct": correct_flag(probs[fact_id], probs[foil_id]),
        f"{prefix}_acc_fact_highest": highest_rank(probs, fact_id),
        f"{prefix}_acc_foil_highest": highest_rank(probs, foil_id),
        f"{prefix}_acc_fact_highest_concept": highest_concept(probs, fact_id, l0_tl, l1_tl),
        f"{prefix}_acc_foil_highest_concept": highest_concept(probs, foil_id, l0_tl, l1_tl),
    }
    
#%% metric aggregators
def compute_all_faith_metrics(base_distribs, P_distribs, I_P_distribs, c_index):
    fth_mis = compute_faith_mi(base_distribs, P_distribs, I_P_distribs, c_index)

    P_fth_kls = compute_faith_kls(base_distribs, P_distribs, prefix="P_")
    I_P_fth_kls = compute_faith_kls(base_distribs, I_P_distribs, prefix="I_P_")

    P_fth_tvds = compute_faith_tvds(base_distribs, P_distribs, prefix="P_")
    I_P_fth_tvds = compute_faith_tvds(base_distribs, I_P_distribs, prefix="I_P_")

    P_fth_pct_chg = compute_faith_pct_chg(base_distribs, P_distribs, prefix="P_")
    I_P_fth_pct_chg = compute_faith_pct_chg(base_distribs, I_P_distribs, prefix="I_P_")
    
    return fth_mis | P_fth_kls | P_fth_tvds | P_fth_pct_chg | I_P_fth_kls\
         | I_P_fth_tvds | I_P_fth_pct_chg

"""    
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
"""
def compute_all_erasure_components(base_distribs, P_distribs, I_P_distribs):
    return compute_all_h_c_h(base_distribs, P_distribs, I_P_distribs)

def compute_all_acc_flags(base_distribs, P_distribs, I_P_distribs, 
    fact_id, foil_id, l0_tl, l1_tl):
    base_flags = compute_factfoil_flags(
        base_distribs["all_split"], 
        fact_id, foil_id, l0_tl, l1_tl, "base"
    )
    P_flags = compute_factfoil_flags(
        P_distribs["all_split"], 
        fact_id, foil_id, l0_tl, l1_tl, "P"
    )
    I_P_flags = compute_factfoil_flags(
        I_P_distribs["all_split"], 
        fact_id, foil_id, l0_tl, l1_tl, "I_P"
    )
    return base_flags | P_flags | I_P_flags

#%% main runners
def compute_eval_one_sample(h, P, I_P, V, l0_tl, l1_tl, c_index, 
    fact_id, foil_id, faith=True, mi=True, acc=True, X_pca=None):
    base_distribs, P_distribs, I_P_distribs = get_all_distribs(
        h, P, I_P, V, l0_tl, l1_tl
    )

    faith_metrics, er_kl_metrics, er_mis_metrics = {},{},{}
    if faith:
        faith_metrics = compute_all_faith_metrics(
            base_distribs, P_distribs, I_P_distribs, c_index
        )
    #if er_kls:
    #    er_kl_metrics = compute_all_erasure_kls(
    #        base_distribs, P_distribs, I_P_distribs, pair_probs
    #    )
    if mi:
        mi_components = compute_all_erasure_components(
            base_distribs, P_distribs, I_P_distribs
        )
    if acc:
        acc_flags = compute_all_acc_flags(
            base_distribs, P_distribs, I_P_distribs, 
            fact_id, foil_id, l0_tl, l1_tl
        )
    return faith_metrics | mi_components | acc_flags #| er_kl_metrics

def compute_eval(hs_wff, P, I_P, V, l0_tl, l1_tl, c_index, 
    faith=True, mi=True, acc=True, X_pca=None):
    #ind = get_hs_sample_index(hs, nsamples)    

    pbar = tqdm(range(len(hs_wff)))
    pbar.set_description("Computing eval on hs")
    metrics_per_sample = []
    for i in pbar:
        h, faid, foid = hs_wff[i]
        metrics_per_sample.append(
            compute_eval_one_sample(
                h, P, I_P, V, l0_tl, l1_tl, c_index, 
                faid, foid, faith=faith, mi=mi, acc=acc, X_pca=X_pca
        ))
    df = pd.DataFrame(metrics_per_sample)
    df["concept_label"] = c_index
    return df

def get_hs_sample_index(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    return idx[:nsamples]

"""
def compute_kls_all_hs(concept_name, model_name, concept_hs, other_hs, P, I_P):
    other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)

    concept_kls = compute_kls(concept_hs, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals)
    other_kls = compute_kls(other_hs, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals)
    
    concept_kls_desc = concept_kls.describe()
    concept_kls_desc.columns = ["concept_" + x for x in concept_kls_desc.columns]
    other_kls_desc = other_kls.describe()
    other_kls_desc.columns = ["other_" + x for x in other_kls_desc.columns]
    
    all_kls = pd.concat([concept_kls, other_kls], axis=0)
    all_kls_desc = all_kls.describe()
    all_kls_desc.columns = ["all_" + x for x in all_kls_desc.columns]
    
    all_descs = pd.concat([concept_kls_desc, other_kls_desc, all_kls_desc], axis=1)
    return all_descs, concept_kls, other_kls
"""

def compute_kls_from_run_output(concept_name, model_name, run_output_path, nsamples):
    #TODO: do this with Xtest now that it is logged to run output
    concept_hs = load_hs(concept_name, model_name, nsamples=nsamples)
    other_hs = load_other_hs(concept_name, model_name, nsamples=nsamples)
    P, I_P = load_run_Ps(run_output_path)

    return compute_kls_all_hs(concept_name, model_name, concept_hs, other_hs, 
        P, I_P)    
    
def compute_kls_after_training(concept_name, model_name, X_test, P, I_P):
    other_hs = load_other_hs(concept_name, model_name, nsamples=X_test.shape[0])
    
    return compute_kls_all_hs(concept_name, model_name, X_test, other_hs, 
        P, I_P)

def compute_eval_filtered_hs(model_name, concept, P, I_P, l0_hs_wff, l1_hs_wff,
    p_x=None):
    V, l0_tl, l1_tl = load_model_eval(model_name, concept)
    
    l0_eval = compute_eval(l0_hs_wff, P, I_P, V, l0_tl, l1_tl, 0)
    l1_eval = compute_eval(l1_hs_wff, P, I_P, V, l0_tl, l1_tl, 1)
    full_eval = pd.concat((l0_eval, l1_eval),axis=0)
    full_eval_means = full_eval.mean()

    h_c = compute_h_c_bin(l0_hs_wff, l1_hs_wff)
    compute_mi(h_c, full_eval_means)
    if p_x is not None:
        compute_h_x(p_x, l0_tl, l1_tl, full_eval_means)
    return full_eval, full_eval_means

#def compute_kls_from_generations(concept_name, model_name, P, I_P, nsamples=100):
#    V, l0_tl, l1_tl, p_c = load_model_eval_ar(concept_name, model_name)
#    l0_hs, l1_hs = load_filtered_hs(model_name, "no_I_P", nsamples)
#    p_c, h_c = get_h_c_bin(model_name, "no_I_P")
#    return compute_kls_all_hs(concept_name, model_name, X_test, other_hs, 
#        P, I_P)


#%%#################
# Main             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Running KL and MI eval')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["gender", "number"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=BERT_LIST + GPT2_LIST,
        help="Models to create embedding files for"
    )
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.info(args)

    #model_name = args.model
    #concept_name = args.concept
    model_name = "gpt2-large"
    concept_name = "number"
    nsamples = 1000
    
    logging.info(f"Running KL and MI eval for {model_name}, {concept_name}")

    outdir = os.path.join(OUT, "raw_results")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, f"kl_mi_{model_name}_{concept_name}.csv")

    run_output_dir = os.path.join(OUT, f"run_output/{concept_name}/{model_name}")
    run_path = os.path.join(run_output_dir, "230614/run_gpt2-large_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-19-13:34:07_0_1.pkl")

    kls = compute_kls_from_run_output(concept_name, model_name, run_path, nsamples)
    kls.to_csv(outfile)
    logging.info(f"Exported KLs for all subsets of hs to {outfile}")
