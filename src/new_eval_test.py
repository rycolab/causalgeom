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
from data.filter_generations import load_filtered_hs_wff

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%%
run_eval_path = os.path.join(RESULTS, "number/gpt2-large/eval_run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")
#run_eval_path = os.path.join(RESULTS, "number/bert-base-uncased/eval_run_bert-base-uncased_theta_k2_Plr0.003_Pms11,21,31,41,51_clflr0.003_clfms200_2023-06-26-22:57:54_0_3.pkl")

with open(run_eval_path, "rb") as f:
    data = pickle.load(f)

#run_info = data["acc_df"].loc[0, ["concept", "model", "k"]]

# %%
#df = data["test_eval_samples"]

def compute_split_metrics(sample_df, sample_origin):
    if sample_df is not None:
        split_metrics = sample_df[
            ["concept_label", "P_fth_mi", "I_P_fth_mi", "P_acc_correct", "I_P_acc_correct"]
        ].groupby(["concept_label"]).mean()
        l0_metrics = split_metrics.loc[0, :]
        l0_metrics.index = [f"{sample_origin}_{x}_l0" for x in l0_metrics.index]
        l1_metrics = split_metrics.loc[1, :]
        l1_metrics.index = [f"{sample_origin}_{x}_l1" for x in l1_metrics.index]
        all_metrics = pd.concat((l0_metrics, l1_metrics), axis=0)
        #all_metrics["origin"] = sample_origin
        return all_metrics
    else:
        return None

def compute_combined_metrics(eval_df, sample_origin):
    combined_metrics = eval_df.loc[["base_mi", "P_mi","I_P_mi", "P_acc_correct", "I_P_acc_correct"]]
    combined_metrics["reconstructed_info"] = combined_metrics["P_mi"] + combined_metrics["I_P_mi"]
    combined_metrics["encapsulation"] = combined_metrics["base_mi"] - combined_metrics["P_mi"]
    combined_metrics["perc_reconstructed"] = combined_metrics["reconstructed_info"] / combined_metrics["base_mi"]
    combined_metrics.index = [f"{sample_origin}_{x}" for x in combined_metrics.index]
    return combined_metrics

def format_metrics(eval_dict, prefix):
    split_metrics = compute_split_metrics(eval_dict[f"{prefix}_eval_samples"], prefix)
    #combined_metrics = eval_dict[f"{prefix}_eval"].loc[["P_mi","I_P_mi", "P_acc_correct", "I_P_acc_correct"]]
    #combined_metrics.index = [f"{prefix}_{x}" for x in combined_metrics.index]
    combined_metrics = compute_combined_metrics(eval_dict[f"{prefix}_eval"], prefix)
    all_metrics = pd.DataFrame(
        pd.concat((split_metrics, combined_metrics), axis=0)).reset_index()
    all_metrics.columns = ["metric", "value"]
    #all_metrics["origin"] = prefix
    return all_metrics

def format_sample_eval(eval_dict):
    run_info = eval_dict["acc_df"].loc[0, ["concept", "model", "k"]].to_dict()
    test_metrics = format_metrics(eval_dict, "test")
    if eval_dict["gen_eval"] is not None:
        gen_metrics = format_metrics(eval_dict, "gen")
        all_metrics = pd.concat(
            [test_metrics, gen_metrics], axis=0).reset_index(drop=True)
    else:
        all_metrics = test_metrics
    all_metrics["model"] = run_info["model"]
    all_metrics["concept"] = run_info["concept"]
    all_metrics["k"] = run_info["k"]
    return all_metrics



#%%
test = format_sample_eval(data)
#%%
combined_metrics = data["test_eval"].loc[["base_mi", "P_mi","I_P_mi", "P_acc_correct", "I_P_acc_correct"]]
combined_metrics["reconstructed_info"] = combined_metrics["P_mi"] + combined_metrics["I_P_mi"]
combined_metrics["encapsulation"] = combined_metrics["base_mi"] - combined_metrics["P_mi"]
combined_metrics["perc_reconstructed"] = combined_metrics["reconstructed_info"] / combined_metrics["base_mi"]

#%%
gen_split_metrics = compute_split_metrics(data["gen_eval_samples"], "gen")
#def format_combined_metrics(test_eval, gen_eval)
#test_combined_metrics["origin"] = "test"
gen_combined_metrics = data["gen_eval"].loc[["P_mi","I_P_mi"]].to_dict()
#gen_combined_metrics["origin"] = "gen"
#return pd.DataFrame.from_records([test_combined_metrics, gen_combined_metrics])

all_metrics = pd.concat([l0_metrics, l1_metrics, combined_metrics], axis=0)

# %%
