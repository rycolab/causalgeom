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
from tqdm import tqdm, trange
import pandas as pd
import pickle

from scipy.special import softmax, kl_div

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS
#from utils.lm_loaders import get_tokenizer, get_V
from evals.kl_eval import load_run_output, get_distribs, \
    normalize_pairs, compute_overall_mi, compute_kl, renormalize
from utils.dataset_loaders import load_hs, load_model_eval

#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Main             #
####################
#if __name__ == '__main__':

model_name = "bert-base-uncased"
concept = "number"
suffix = "nopca"
gpt_run_output = os.path.join(OUT, "run_output/linzen/gpt2/230415/run_gpt2_k1_Pms11,16,21,26,31,36_Pg0.5_clfms21,31_clfg0.5_2023-04-15-15:02:02_0_1.pkl")
gpt2_large_run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")
bert_run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")
if model_name == "bert-base-uncased":
    run_output = bert_run_output
elif model_name == "gpt2":
    run_output = gpt_run_output
elif model_name == "gpt2-large":
    run_output = gpt2_large_run_output
elif model_name == "gpt2-medium":
    run_output = None
else:
    run_output = None

RES = pd.read_csv(os.path.join(OUT, "run_kls.csv"), index_col=0)
OUTDIR = os.path.join(RESULTS, f"{concept}/{model_name}")
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
logging.info(f"Formatting and exporting run results for {model_name}")

#P, I_P = load_run_output(gpt_run_output)

#kls = compute_kls(hs, P, I_P, word_emb, l0_emb, l1_emb, verb_probs)
#kls.to_csv(os.path.join(OUT, "run_kls.csv"))


#%%#################
# TABLE MAKING.    #
####################
#with open(run_output, 'rb') as f:      
#    run = pickle.load(f)

#%% Diag eval
"""
diag_acc_keys = [
    "diag_acc_original_test", "diag_loss_original_test", 
    "diag_acc_P_burn_test", "diag_loss_P_burn_test",
    "diag_acc_I_P_burn_test", "diag_loss_I_P_burn_test"
]

usage_acc_keys = [
    "lm_acc_original_test", "lm_loss_original_test", 
    "lm_acc_P_burn_test", "lm_loss_P_burn_test",
    "lm_acc_I_P_burn_test", "lm_loss_I_P_burn_test"
]

diag_usage_res = dict(
    orig=dict(
        test_diag_loss=run["diag_eval"]["diag_loss_original_test"],
        test_diag_acc=run["diag_eval"]["diag_acc_original_test"],
        test_lm_loss=run["usage_eval"]["lm_loss_original_test"],
        test_lm_acc=run["usage_eval"]["lm_acc_original_test"],
    ),
    P=dict(
        test_diag_loss=run["diag_eval"]["diag_loss_P_burn_test"],
        test_diag_acc=run["diag_eval"]["diag_acc_P_burn_test"],
        test_lm_loss=run["usage_eval"]["lm_loss_P_burn_test"],
        test_lm_acc=run["usage_eval"]["lm_acc_P_burn_test"],
    ),
    I_P=dict(
        test_diag_loss=run["diag_eval"]["diag_loss_I_P_burn_test"],
        test_diag_acc=run["diag_eval"]["diag_acc_I_P_burn_test"],
        test_lm_loss=run["usage_eval"]["lm_loss_I_P_burn_test"],
        test_lm_acc=run["usage_eval"]["lm_acc_I_P_burn_test"],
    ),
)

diag_usage_res_df = pd.DataFrame(diag_usage_res).T
diag_usage_res_path = os.path.join(OUTDIR, f"diag_usage_res_{suffix}.csv")
diag_usage_res_df.to_csv(diag_usage_res_path)
logging.info(f"Exported diag_usage results to: {diag_usage_res_path}")
"""
#%%
def get_fth_res(res, split, P, metric):
    """
    res : pd.DataFrame.describe() output 
    split: "concept", "other", "all"
    P: "P", "I_P"
    metric: "mean", "std"
    """
    resdict = dict(
        kl=dict(
            all_split=res.loc[metric, f"{split}_{P}_faith_kl_all_split"],
            all_merged=res.loc[metric, f"{split}_{P}_faith_kl_all_merged"],
            tgt_split=res.loc[metric, f"{split}_{P}_faith_kl_tgt_split"],
            tgt_merged=res.loc[metric, f"{split}_{P}_faith_kl_tgt_merged"],
            other=res.loc[metric, f"{split}_{P}_faith_kl_other"],
        ),
        tvd=dict(
            all_split=res.loc[metric, f"{split}_{P}_faith_tvd_all_split"],
            all_merged=res.loc[metric, f"{split}_{P}_faith_tvd_all_merged"],
            tgt_split=res.loc[metric, f"{split}_{P}_faith_tvd_tgt_split"],
            tgt_merged=res.loc[metric, f"{split}_{P}_faith_tvd_tgt_merged"],
            other=res.loc[metric, f"{split}_{P}_faith_tvd_other"],
        ),
        pct_chg=dict(
            all_split=res.loc[metric, f"{split}_{P}_faith_pct_chg_all_split"],
            all_merged=res.loc[metric, f"{split}_{P}_faith_pct_chg_all_merged"],
            tgt_split=res.loc[metric, f"{split}_{P}_faith_pct_chg_tgt_split"],
            tgt_merged=res.loc[metric, f"{split}_{P}_faith_pct_chg_tgt_merged"],
            other=res.loc[metric, f"{split}_{P}_faith_pct_chg_other"],
        ),
    )
    df = pd.DataFrame(resdict).T.reset_index(names="distance_metric")
    df["split"] = split
    df["metric"] = metric
    df = df[["split", "distance_metric", "metric"] + [col for col in df.columns if col not in ["split", "metric", "distance_metric"]]]
    return df

def get_full_kls_df(res, P):
    """ P: "P", "I_P" """
    concept_P_mean = get_fth_res(res, "concept", P, "mean")
    concept_P_std = get_fth_res(res, "concept", P, "std")

    other_P_mean = get_fth_res(res, "other", P, "mean")
    other_P_std = get_fth_res(res, "other", P, "std")

    all_P_mean = get_fth_res(res, "all", P, "mean")
    all_P_std = get_fth_res(res, "all", P, "std")

    full_P_kls = pd.concat([concept_P_mean, concept_P_std, other_P_mean, other_P_std, all_P_mean, all_P_std], axis=0)
    full_P_kls.sort_values(by = ["split", "distance_metric", "metric"], inplace=True)
    return full_P_kls 

P_kls = get_full_kls_df(RES, "P")
I_P_kls = get_full_kls_df(RES, "I_P")

P_fth_res_path = os.path.join(OUTDIR, f"fth_res_P_{suffix}.csv")
P_kls.to_csv(P_fth_res_path)
logging.info(f"Exported P_fth results to: {P_fth_res_path}")

I_P_fth_res_path = os.path.join(OUTDIR, f"fth_res_I_P_{suffix}.csv")
I_P_kls.to_csv(I_P_fth_res_path)
logging.info(f"Exported I_P_fth results to: {I_P_fth_res_path}")

#%%
def get_er_res(res, split, metric):
    resdict = dict(
        base=dict(
            overall_mi=res.loc[metric, f"{split}_base_overall_mi"],
            lemma_mi=res.loc[metric, f"{split}_base_lemma_mi"],
            pairwise_mi=res.loc[metric, f"{split}_base_pairwise_mi"],
        ),
        P=dict(
            overall_mi=res.loc[metric, f"{split}_P_overall_mi"],
            lemma_mi=res.loc[metric, f"{split}_P_lemma_mi"],
            pairwise_mi=res.loc[metric, f"{split}_P_pairwise_mi"],
        ),
        I_P=dict(
            overall_mi=res.loc[metric, f"{split}_I_P_overall_mi"],
            lemma_mi=res.loc[metric, f"{split}_I_P_lemma_mi"],
            pairwise_mi=res.loc[metric, f"{split}_I_P_pairwise_mi"],
        ),
    )
    df = pd.DataFrame(resdict).T.reset_index(names="reps")
    df["split"] = split
    df["metric"] = metric
    df = df[["split", "reps", "metric"] + [col for col in df.columns if col not in ["split", "metric", "reps"]]]
    return df

def get_full_er_df(res):
    concept_mean = get_er_res(res, "concept", "mean")
    concept_std = get_er_res(res, "concept", "std")

    other_mean = get_er_res(res, "other", "mean")
    other_std = get_er_res(res, "other", "std")

    all_mean = get_er_res(res, "all", "mean")
    all_std = get_er_res(res, "all", "std")

    full_ers = pd.concat(
        [concept_mean, concept_std, other_mean, other_std, 
            all_mean, all_std], axis=0)
    full_ers.sort_values(by = ["split", "reps", "metric"], inplace=True)
    return full_ers

full_ers = get_full_er_df(RES) 

er_res_path = os.path.join(OUTDIR, f"er_res_{suffix}.csv")
full_ers.to_csv(er_res_path)
logging.info(f"Exported erasure results to: {er_res_path}")

#%%#################
# BASELINE       #
####################
nsamples = 200

hs_sub = load_hs(concept, model_name, nsamples*2)
other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept, model_name)

kls = []
for i in trange(nsamples):
    h1 = hs_sub[i]
    h2 = hs_sub[nsamples + i]
    h1_base_distribs = get_distribs(h1, other_emb, l0_emb, l1_emb)
    h2_base_distribs = get_distribs(h2, other_emb, l0_emb, l1_emb)
    res = dict(
        all_split=compute_kl(h1_base_distribs["all_split"], h2_base_distribs["all_split"]),
        all_merged=compute_kl(h1_base_distribs["all_merged"], h2_base_distribs["all_merged"]),
        tgt_split=compute_kl(renormalize(h1_base_distribs["lemma_split"]), renormalize(h2_base_distribs["lemma_split"])),
        tgt_merged=compute_kl(renormalize(h1_base_distribs["lemma_merged"]), renormalize(h2_base_distribs["lemma_merged"])),
        other=compute_kl(renormalize(h1_base_distribs["other"]), renormalize(h2_base_distribs["other"])),
    )
    kls.append(res)
    
desc_kls = pd.DataFrame(kls).describe()
desc_kls_path = os.path.join(OUTDIR, "fth_baseline.csv")
desc_kls.to_csv(desc_kls_path)
logging.info(f"Exported baseline KL results to: {desc_kls_path}")