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
    SG_PL_PROB = os.path.join(DATASETS, "processed/linzen_word_lists/sg_pl_prob.pkl")

    word_emb = np.load(WORD_EMB)
    verb_p = np.load(VERB_P)
    sg_emb = np.load(SG_EMB)
    pl_emb = np.load(PL_EMB)
    with open(SG_PL_PROB, 'rb') as f:      
        sg_pl_prob = pickle.load(f).to_numpy()
    return word_emb, sg_emb, pl_emb, verb_p, sg_pl_prob

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

def get_all_pairwise_distribs(base_distribs, P_distribs, I_P_distribs):
    base_pair_probs = normalize_pairs(base_distribs["sg"], base_distribs["pl"])
    P_pair_probs = normalize_pairs(P_distribs["sg"], P_distribs["pl"])
    I_P_pair_probs = normalize_pairs(I_P_distribs["sg"], I_P_distribs["pl"])
    return base_pair_probs, P_pair_probs, I_P_pair_probs

def get_distribs(h, word_emb, sg_emb, pl_emb):
    all_split, words, sg, pl = get_probs(h, word_emb, sg_emb, pl_emb)
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

def get_all_distribs(h, P, I_P, word_emb, sg_emb, pl_emb):
    base_distribs = get_distribs(h, word_emb, sg_emb, pl_emb)
    P_distribs = get_distribs(P @ h, word_emb, sg_emb, pl_emb)
    I_P_distribs = get_distribs(I_P @ h, word_emb, sg_emb, pl_emb)
    return base_distribs, P_distribs, I_P_distribs

def compute_kl(p, q, agg_func=np.sum):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        return 0
    else:
        return agg_func(kl_div(p, q))

def renormalize(p):
    return p / np.sum(p)

def compute_faith_kls(base, proj, prefix="", agg_func=np.sum):
    kls = {
        f"{prefix}faith_kl_all_split": compute_kl(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_kl_all_merged": compute_kl(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_kl_words": compute_kl(renormalize(base["words"]), renormalize(proj["words"]), agg_func),
        f"{prefix}faith_kl_tgt_split": compute_kl(renormalize(base["lemma_split"]), renormalize(proj["lemma_split"]), agg_func),
        f"{prefix}faith_kl_tgt_merged": compute_kl(renormalize(base["lemma_merged"]), renormalize(proj["lemma_merged"]), agg_func),
        f"{prefix}faith_kl_words_unnorm": compute_kl(base["words"], proj["words"], agg_func),
        f"{prefix}faith_kl_tgt_split_unnorm": compute_kl(base["lemma_split"], proj["lemma_split"], agg_func),
    }
    return kls
    
def compute_tvd(p, q, agg_func=np.mean):
    if not (np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1)):
        logging.warn("Distribution not normalized before KL")
        return 0
    else:
        return agg_func(np.abs(p-q))
    
def compute_faith_tvds(base, proj, prefix="", agg_func=np.mean):
    tvds = {
        f"{prefix}faith_tvd_all_split": compute_tvd(base["all_split"], proj["all_split"], agg_func),
        f"{prefix}faith_tvd_all_merged": compute_tvd(base["all_merged"], proj["all_merged"], agg_func),
        f"{prefix}faith_tvd_words": compute_tvd(renormalize(base["words"]), renormalize(proj["words"]), agg_func),
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
        f"{prefix}faith_pct_chg_words": compute_pct_chg(renormalize(base["words"]), renormalize(proj["words"]), agg_func),
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

def compute_erasure_kls(base_pair_probs, proj_pair_probs, verb_probs, prefix="", agg_func=np.sum):
    erasure_kls = {
        f"{prefix}er_kl_base_proj": compute_erasure_kl(
            base_pair_probs, proj_pair_probs, agg_func),
        f"{prefix}er_kl_maj_base": compute_erasure_kl(
            verb_probs, base_pair_probs, agg_func),
        f"{prefix}er_kl_maj_proj": compute_erasure_kl(
            verb_probs, proj_pair_probs, agg_func)
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
def get_sg_pl_prob(sg_prob, pl_prob):
    total_sg_prob = np.sum(sg_prob)
    total_pl_prob = np.sum(pl_prob)
    total_prob = total_sg_prob + total_pl_prob
    sg_pl_prob = np.hstack([total_sg_prob, total_pl_prob]) / total_prob
    return sg_pl_prob

def compute_overall_mi(uncond_sg_pl_prob, cond_sg_probs, cond_pl_probs):
    cond_sg_pl_prob = get_sg_pl_prob(cond_sg_probs, cond_pl_probs)
    return entropy(uncond_sg_pl_prob) - entropy(cond_sg_pl_prob)

def get_all_overall_mis(uncond_sg_pl_prob, base_distribs, P_distribs, I_P_distribs):
    res = dict(
        base_overall_mi = compute_overall_mi(
            uncond_sg_pl_prob, base_distribs["sg"], base_distribs["pl"]),
        P_overall_mi = compute_overall_mi(
            uncond_sg_pl_prob, P_distribs["sg"], P_distribs["pl"]),
        I_P_overall_mi = compute_overall_mi(
            uncond_sg_pl_prob, I_P_distribs["sg"], I_P_distribs["pl"]),
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

    
def compute_all_erasure_kls(base_distribs, P_distribs, I_P_distribs, verb_probs):
    base_pair_probs, P_pair_probs, I_P_pair_probs = get_all_pairwise_distribs(
        base_distribs, P_distribs, I_P_distribs
    )
    
    P_er_kls = compute_erasure_kls(
        base_pair_probs, P_pair_probs, verb_probs, prefix="P_"
    )
    I_P_er_kls = compute_erasure_kls(
        base_pair_probs, I_P_pair_probs, verb_probs, prefix="I_P_"
    )
    return P_er_kls | I_P_er_kls

def compute_all_erasure_mis(base_distribs, P_distribs, I_P_distribs, verb_probs, sg_pl_probs):
    base_pair_probs, P_pair_probs, I_P_pair_probs = get_all_pairwise_distribs(
        base_distribs, P_distribs, I_P_distribs
    )
    
    pairwise_mis = get_all_pairwise_mis(
        verb_probs, base_pair_probs, P_pair_probs, I_P_pair_probs
    )
    overall_mis = get_all_overall_mis(
        sg_pl_probs, base_distribs, P_distribs, I_P_distribs
    )
    return pairwise_mis | overall_mis

def compute_kls_one_sample(h, P, I_P, word_emb, sg_emb, pl_emb, verb_probs, 
    sg_pl_probs, faith=True, er_kls=True, er_mis=True):
    base_distribs, P_distribs, I_P_distribs = get_all_distribs(
        h, P, I_P, word_emb, sg_emb, pl_emb
    )

    faith_metrics, er_kl_metrics, er_mis_metrics = {},{},{}
    if faith:
        faith_metrics = compute_all_faith_metrics(
            base_distribs, P_distribs, I_P_distribs
        )
    if er_kls:
        er_kl_metrics = compute_all_erasure_kls(
            base_distribs, P_distribs, I_P_distribs, verb_probs
        )
    if er_mis:
        er_mis_metrics = compute_all_erasure_mis(
            base_distribs, P_distribs, I_P_distribs, verb_probs, sg_pl_probs
        )
    return faith_metrics | er_kl_metrics | er_mis_metrics

def get_hs_sample_index(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    return idx[:nsamples]

def compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob, 
    nsamples=200, faith=True, er_kls=True, er_mis=True):
    ind = get_hs_sample_index(hs, nsamples)    

    pbar = tqdm(ind)
    pbar.set_description("Computing faithfulness and erasure KL on hidden states")
    kls = []
    for i in pbar:
        kls.append(compute_kls_one_sample(
            hs[i], P, I_P, word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob,
            faith, er_kls, er_mis
        ))
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
    word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob = load_model_eval(model_name)
    P, I_P = load_run_output(run_output)
    
    kls = compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob)
    kls.to_csv(os.path.join(OUT, "run_kls.csv"))
