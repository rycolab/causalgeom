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
from data.filter_generations import load_filtered_hs, load_filtered_hs_wff
from test_eval import filter_test_hs_wff, create_er_df, create_fth_df, \
    compute_kl_baseline
from evals.kl_eval import get_distribs, correct_flag, highest_rank, highest_concept
from tqdm import tqdm


coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
def get_inj_vectors(V, l0_tl, l1_tl, P):
    V0 = V[l0_tl]
    V1 = V[l1_tl]

    V0proj = V0 @ P
    V1proj = V1 @ P

    V0_mean = np.mean(V0proj, axis=0)
    V1_mean = np.mean(V1proj, axis=0)

    V0meannorm = V0_mean / np.linalg.norm(V0_mean)
    V1meannorm = V1_mean / np.linalg.norm(V1_mean)
    return V0meannorm, V1meannorm

def get_inj_accs(hs_wff, case, V, l0_tl, l1_tl, P, I_P, V0inj, V1inj, alpha):
    Phn = []
    reslist = []
    for h, faid, foid in tqdm(hs_wff):
        base_distribs = get_distribs(h, V, l0_tl, l1_tl)
        I_P_distribs = get_distribs(h.T @ I_P, V, l0_tl, l1_tl)
        normPh = np.linalg.norm(h.T @ P)
        Phn.append(normPh)
        I_P_inj0_distribs = get_distribs(
            (h.T @ I_P) + (V0inj * alpha * normPh), V, l0_tl, l1_tl
        )
        I_P_inj1_distribs = get_distribs(
            (h.T @ I_P) + (V1inj * alpha * normPh), V, l0_tl, l1_tl
        )
        if case == 0:
            l0id, l1id = faid, foid
        else:
            l0id, l1id = foid, faid
        reslist.append(dict(
            base_correct = correct_flag(base_distribs["all_split"][faid], base_distribs["all_split"][foid]),
            base_correct_highest = highest_rank(base_distribs["all_split"], faid),
            base_correct_highest_concept = highest_concept(base_distribs["all_split"], faid, l0_tl, l1_tl),
            I_P_correct = correct_flag(I_P_distribs["all_split"][faid], I_P_distribs["all_split"][foid]),
            I_P_l0_highest = highest_rank(I_P_distribs["all_split"], l0id),
            I_P_l1_highest = highest_rank(I_P_distribs["all_split"], l1id),
            I_P_l0_highest_concept = highest_concept(I_P_distribs["all_split"], l0id, l0_tl, l1_tl),
            I_P_l1_highest_concept = highest_concept(I_P_distribs["all_split"], l1id, l0_tl, l1_tl),
            I_P_inj0_correct = correct_flag(I_P_inj0_distribs["all_split"][l0id], I_P_inj0_distribs["all_split"][l1id]),
            I_P_inj0_l0_highest = highest_rank(I_P_inj0_distribs["all_split"], l0id),
            I_P_inj0_l1_highest = highest_rank(I_P_inj0_distribs["all_split"], l1id),
            I_P_inj0_l0_highest_concept = highest_concept(I_P_inj0_distribs["all_split"], l0id, l0_tl, l1_tl),
            I_P_inj0_l1_highest_concept = highest_concept(I_P_inj0_distribs["all_split"], l1id, l0_tl, l1_tl),
            I_P_inj1_correct = correct_flag(I_P_inj1_distribs["all_split"][l1id], I_P_inj1_distribs["all_split"][l0id]),
            I_P_inj1_l0_highest = highest_rank(I_P_inj1_distribs["all_split"], l0id),
            I_P_inj1_l1_highest = highest_rank(I_P_inj1_distribs["all_split"], l1id),
            I_P_inj1_l0_highest_concept = highest_concept(I_P_inj1_distribs["all_split"], l0id, l0_tl, l1_tl),
            I_P_inj1_l1_highest_concept = highest_concept(I_P_inj1_distribs["all_split"], l1id, l0_tl, l1_tl),
        ))
    return reslist, Phn

def compute_inj_eval_run(model_name, concept, run, run_path, nsamples, alpha):
    P, I_P = load_run_Ps(run_path)
    V, l0_tl, l1_tl, _ = load_model_eval(model_name, concept)
    test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
    run["X_test"], run["facts_test"], run["foils_test"], 
        l0_tl, l1_tl, nsamples=nsamples
    )

    V0inj, V1inj = get_inj_vectors(V, l0_tl, l1_tl, P) 
    l0_reslist, l0_Phn = get_inj_accs(
        test_l0_hs_wff, 0, V, l0_tl, l1_tl, P, I_P, V0inj, V1inj, alpha
    )
    l1_reslist, l1_Phn = get_inj_accs(
        test_l1_hs_wff, 1, V, l0_tl, l1_tl, P, I_P, V0inj, V1inj, alpha
    )

    #l0_means = pd.DataFrame(l0_reslist).mean()
    #l1_means = pd.DataFrame(l1_reslist).mean()
    #all_means = pd.DataFrame(l0_reslist + l1_reslist).mean()
    #combo_df = pd.concat((l0_means, l1_means, all_means), axis=1)
    #combo_df.columns = ["l0_means", "l1_means", "all_means"]
    #return combo_df
    l0_df = pd.DataFrame(l0_reslist)
    l0_df["y"] = 0
    l1_df = pd.DataFrame(l1_reslist)
    l1_df["y"] = 1
    all_df = pd.concat((l0_df, l1_df), axis=0)
    return all_df



def eval_handler_pair(model_name, concept, run_output_folder, nsamples, alpha):
    rundir = os.path.join(OUT, f"run_output/{concept}/{model_name}/{run_output_folder}")
    outdir = os.path.join(RESULTS, "inj")
    #outdir = RESULTS
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    runs = [x for x in os.listdir(rundir) if x.endswith(".pkl")]

    for run in runs:
        run_path = os.path.join(rundir, run)
        outpath = os.path.join(outdir, f"{concept}_{model_name}_injacc_{run[:-4]}.csv")

        run = load_run_output(run_path)
        if run["config"]["k"] != 1:
            continue
        elif os.path.exists(outpath):
            logging.info(f"Run already evaluated: {run_path}")
            continue
        else:
            inj_accs_df = compute_inj_eval_run(
                model_name, concept, run, run_path, nsamples, alpha
            )
            inj_accs_df.to_csv(outpath)
            #with open(outpath, "wb") as f:
            #    pickle.dump(run_eval_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Run eval exported: {run_path}")

    logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}")

def compute_inj_eval_pairs(pairs, nsamples, alpha):
    for model_name, concept, run_output_folder in pairs:
        eval_handler_pair(model_name, concept, run_output_folder, nsamples, alpha)
    logging.info("Finished computing all pairs of evals")


#%% diagnostics
#from sklearn.metrics.pairwise import cosine_similarity
#from scipy import sparse
#from scipy.spatial.distance import cdist
#import seaborn as sns
#A =  np.vstack([V0[:10], V1[:10], V0_mean.reshape(1,-1), V1_mean.reshape(1,-1)])
#sns.heatmap(cdist(A, A, metric = "cosine"))

#%%#################
# Main             #
####################
if __name__=="__main__":
    pairs = [
        ("gpt2-large", "number", "230627"),
        ("bert-base-uncased", "number", "230627"),
        ("gpt2-base-french", "gender", "230627"),
        ("camembert-base", "gender", "230627"),
        ("gpt2-large", "number", "230627_fix"),
        ("bert-base-uncased", "number", "230627_fix"),
        ("gpt2-base-french", "gender", "230627_fix"),
        ("camembert-base", "gender", "230627_fix"),
        ("gpt2-large", "number", "230628"),
    ]
    alpha = 1
    nsamples = 500

    compute_inj_eval_pairs(pairs, nsamples, alpha)