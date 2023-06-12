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
import random 
#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT


from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%% NEW EVAL
res = pd.read_csv("../out/run_kls.csv", index_col=0)

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

P_kls = get_full_kls_df(res, "P")
I_P_kls = get_full_kls_df(res, "I_P")

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

full_ers = get_full_ers(res) 
#er_res_path = os.path.join(OUT, f"results/{dataset_name}/{model_name}/er_res_{suffix}.csv")
#er_res_df.to_csv(er_res_path)
#logging.info(f"Exported erasure results to: {er_res_path}")


#%%#################
# Computing new MI #
####################
from utils.dataset_loaders import load_processed_data
from scipy.special import softmax, kl_div
from scipy.stats import entropy
from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from utils.dataset_loaders import load_hs, load_model_eval
from evals.kl_eval import load_run_output

model_name = "gpt2-large" #"bert-base-uncased"
concept_name = "number"
#run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")
run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

hs = load_hs(concept_name, model_name)
other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)
P, I_P = load_run_output(run_output)

#%%
from evals.kl_eval import get_all_distribs, get_all_marginals, get_lemma_marginals
from evals.kl_eval import compute_kls_one_sample
h = hs[13]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, other_emb, l0_emb, l1_emb
)

distrib = base_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)

distrib = I_P_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)


#compute_kls_one_sample(h, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals )
# %%
from evals.kl_eval import compute_overall_mi
all_marginals = [
    concept_marginals["p_0_incl_other"], 
    concept_marginals["p_1_incl_other"], 
    concept_marginals["p_other_incl_other"]
]

compute_overall_mi(concept_marginals, distrib["l0"], distrib["l1"], distrib["other"])

#%%#################
# Camembert        #
####################
model = get_model("camembert-base")
tokenizer = get_tokenizer("camembert-base")
print(model)

inputs = tokenizer("La capitale de la France est <mask>", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)

# %%
from data.dataset_loaders import load_model_eval

#model_name = "gpt2-base-french"
#dataset_name = "ud_fr_gsd"
model_name = "gpt2"
dataset_name = "linzen"
model = get_model(model_name)
word_emb, l0_emb, l1_emb, lemma_prob, concept_prob = load_model_eval(dataset_name, model_name)
#%%
from scipy.stats import entropy

entropy(concept_prob)