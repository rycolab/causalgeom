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
#from utils.lm_loaders import get_tokenizer, get_V
from evals.kl_eval import load_hs, load_model_eval, load_run_output,\
    get_distribs, normalize_pairs, compute_overall_mi, compute_kl, renormalize, \
        sample_hs

#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Main             #
####################
#if __name__ == '__main__':

model_name = "bert-base-uncased"
dataset_name = "linzen"
gpt_run_output = os.path.join(OUT, "run_output/gpt2/230310/run_gpt2_k_1_0_1.pkl")
bert_run_output = os.path.join(OUT, "run_output/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")
if model_name == "bert-base-uncased":
    run_output = bert_run_output
elif model_name == "gpt2":
    run_output = gpt_run_output
else:
    run_output = None

logging.info(f"Formatting and exporting run results for {model_name}")

hs = load_hs(dataset_name, model_name)
word_emb, sg_emb, pl_emb, verb_probs, sg_pl_probs = load_model_eval(model_name)

#P, I_P = load_run_output(gpt_run_output)

#kls = compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs)
#kls.to_csv(os.path.join(OUT, "run_kls.csv"))


#%%#################
# TABLE MAKING.    #
####################
with open(run_output, 'rb') as f:      
    run = pickle.load(f)

#%% Diag eval
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

res1 = dict(
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

res1df = pd.DataFrame(res1).T
res1df.to_csv(os.path.join(OUT, f"results/{model_name}/diag_usage_eval.csv"))


#%%
P_fth_res = dict(
    kl=dict(
        all_split=run["burn_kl_eval"]["P_faith_kl_all_split"],
        all_merged=run["burn_kl_eval"]["P_faith_kl_all_merged"],
        tgt_split=run["burn_kl_eval"]["P_faith_kl_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["P_faith_kl_tgt_merged"],
        other=run["burn_kl_eval"]["P_faith_kl_words"],
    ),
    tvd=dict(
        all_split=run["burn_kl_eval"]["P_faith_tvd_all_split"],
        all_merged=run["burn_kl_eval"]["P_faith_tvd_all_merged"],
        tgt_split=run["burn_kl_eval"]["P_faith_tvd_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["P_faith_tvd_tgt_merged"],
        other=run["burn_kl_eval"]["P_faith_tvd_words"],
    ),
    pct_chg=dict(
        all_split=run["burn_kl_eval"]["P_faith_pct_chg_all_split"],
        all_merged=run["burn_kl_eval"]["P_faith_pct_chg_all_merged"],
        tgt_split=run["burn_kl_eval"]["P_faith_pct_chg_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["P_faith_pct_chg_tgt_merged"],
        other=run["burn_kl_eval"]["P_faith_pct_chg_words"],
    ),
)
 
P_fth_res_df = pd.DataFrame(P_fth_res).T
P_fth_res_df.to_csv(os.path.join(OUT, f"results/{model_name}/fth_res_P.csv"))

#%%
I_P_fth_res = dict(
    kl=dict(
        all_split=run["burn_kl_eval"]["I_P_faith_kl_all_split"],
        all_merged=run["burn_kl_eval"]["I_P_faith_kl_all_merged"],
        tgt_split=run["burn_kl_eval"]["I_P_faith_kl_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["I_P_faith_kl_tgt_merged"],
        other=run["burn_kl_eval"]["I_P_faith_kl_words"],
    ),
    tvd=dict(
        all_split=run["burn_kl_eval"]["I_P_faith_tvd_all_split"],
        all_merged=run["burn_kl_eval"]["I_P_faith_tvd_all_merged"],
        tgt_split=run["burn_kl_eval"]["I_P_faith_tvd_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["I_P_faith_tvd_tgt_merged"],
        other=run["burn_kl_eval"]["I_P_faith_tvd_words"],
    ),
    pct_chg=dict(
        all_split=run["burn_kl_eval"]["I_P_faith_pct_chg_all_split"],
        all_merged=run["burn_kl_eval"]["I_P_faith_pct_chg_all_merged"],
        tgt_split=run["burn_kl_eval"]["I_P_faith_pct_chg_tgt_split"],
        tgt_merged=run["burn_kl_eval"]["I_P_faith_pct_chg_tgt_merged"],
        other=run["burn_kl_eval"]["I_P_faith_pct_chg_words"],
    ),
)
I_P_fth_res_df = pd.DataFrame(I_P_fth_res).T
I_P_fth_res_df.to_csv(os.path.join(OUT, f"results/{model_name}/fth_res_I_P.csv"))

#%%
er_res = dict(
    base=dict(
        overall_mi=run["burn_kl_eval"]["base_overall_mi"],
        pairwise_mi=run["burn_kl_eval"]["base_pairwise_mi"],
    ),
    P=dict(
        overall_mi=run["burn_kl_eval"]["P_overall_mi"],
        pairwise_mi=run["burn_kl_eval"]["P_pairwise_mi"],
    ),
    I_P=dict(
        overall_mi=run["burn_kl_eval"]["I_P_overall_mi"],
        pairwise_mi=run["burn_kl_eval"]["I_P_pairwise_mi"],
    ),
)

er_res_df = pd.DataFrame(er_res).T
er_res_df.to_csv(os.path.join(OUT, f"results/{model_name}/er_res.csv"))


#%%#################
# BASELINE       #
####################
nsamples = 200

hs_sub = sample_hs(hs, nsamples*2)
word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob = load_model_eval(model_name)

kls = []
for i in range(nsamples):
    h1 = hs_sub[i]
    h2 = hs_sub[nsamples + i]
    h1_base_distribs = get_distribs(h1, word_emb, sg_emb, pl_emb)
    h2_base_distribs = get_distribs(h2, word_emb, sg_emb, pl_emb)
    res = dict(
        all_split=compute_kl(h1_base_distribs["all_split"], h2_base_distribs["all_split"]),
        all_merged=compute_kl(h1_base_distribs["all_merged"], h2_base_distribs["all_merged"]),
        tgt_split=compute_kl(renormalize(h1_base_distribs["lemma_split"]), renormalize(h2_base_distribs["lemma_split"])),
        tgt_merged=compute_kl(renormalize(h1_base_distribs["lemma_merged"]), renormalize(h2_base_distribs["lemma_merged"])),
        other=compute_kl(renormalize(h1_base_distribs["words"]), renormalize(h2_base_distribs["words"])),
    )
    kls.append(res)
    
desc_kls = pd.DataFrame(kls).describe()
desc_kls.to_csv(os.path.join(OUT,f"results/{model_name}/fth_baseline.csv"))
