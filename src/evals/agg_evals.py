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
#from scipy.special import softmax
#from scipy.stats import entropy
from tqdm import trange

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
#from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
#from data.embed_wordlists.embedder import load_concept_token_lists
#from evals.kl_eval import load_run_Ps, load_run_output, \
#    compute_eval_filtered_hs, load_model_eval, compute_kl, \
#        renormalize, get_distribs

#from analysis.format_res import get_best_runs
#from data.filter_generations import load_filtered_hs, load_filtered_hs_wff

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

METRIC_MAPPING = {
    'cont_mi':'cont_mi', 
    'I_P_fth_mi': 'stab_mi',
    'I_P_mi': 'mi_c_hbot',
    'P_mi': 'mi_c_hpar', 
    'base_mi': 'mi_c_h', 
    'reconstructed': 'reconstructed', 
    'encapsulation': 'encapsulation', 
    'perc_I_P_mi': 'perc_mi_c_hbot', 
    'perc_P_mi': 'perc_mi_c_hpar',
    'perc_encapsulation': 'perc_encapsulation', 
    'perc_reconstructed': 'perc_reconstructed' 
}
  

#%%
def compute_split_metrics(sample_df, sample_origin):
    if sample_df is not None:
        split_metrics = sample_df[
            #["concept_label", "h_p_x_Ph", "h_p_x_I_Ph", "P_fth_mi", "I_P_fth_mi", "P_acc_correct", "I_P_acc_correct"]
            ["concept_label", "h_p_x_Ph", "h_p_x_I_Ph"]
        ].groupby(["concept_label"]).mean()
        l0_metrics = split_metrics.loc[0, :]
        l0_metrics.index = [f"{x}_l0" for x in l0_metrics.index]
        l1_metrics = split_metrics.loc[1, :]
        l1_metrics.index = [f"{x}_l1" for x in l1_metrics.index]
        all_metrics = pd.concat((l0_metrics, l1_metrics), axis=0)
        #all_metrics["origin"] = sample_origin
        return all_metrics
    else:
        return None

def compute_combined_metrics(eval_df, sample_origin):
    combined_metrics = eval_df.loc[
        ["base_mi", "P_mi","I_P_mi", "P_fth_mi", "I_P_fth_mi", "h_p_x_l0", "h_p_x_l1", "p_c_l0", "p_c_l1"]
    ]
    #combined_metrics = eval_df.loc[["h_p_x_l0", "h_p_x_l1", "P_acc_correct", "I_P_acc_correct"]]
    combined_metrics["reconstructed"] = combined_metrics["P_mi"] + combined_metrics["I_P_mi"]
    combined_metrics["encapsulation"] = combined_metrics["base_mi"] - combined_metrics["P_mi"]
    #percentages
    combined_metrics["perc_P_mi"] = combined_metrics["P_mi"] / combined_metrics["base_mi"]
    combined_metrics["perc_I_P_mi"] = combined_metrics["I_P_mi"] / combined_metrics["base_mi"]
    combined_metrics["perc_encapsulation"] = combined_metrics["encapsulation"] / combined_metrics["base_mi"]
    combined_metrics["perc_reconstructed"] = combined_metrics["reconstructed"] / combined_metrics["base_mi"]
    #combined_metrics.index = [f"{sample_origin}_{x}" for x in combined_metrics.index]
    return combined_metrics

def compute_containment(base_metrics):
    mi_x_Ph_l0 = base_metrics.loc[f"h_p_x_l0"] - base_metrics.loc[f"h_p_x_Ph_l0"]
    mi_x_I_Ph_l0 = base_metrics.loc[f"h_p_x_l0"] - base_metrics.loc[f"h_p_x_I_Ph_l0"]
    mi_x_Ph_l1 = base_metrics.loc[f"h_p_x_l1"] - base_metrics.loc[f"h_p_x_Ph_l1"]
    mi_x_I_Ph_l1 = base_metrics.loc[f"h_p_x_l1"] - base_metrics.loc[f"h_p_x_I_Ph_l1"]

    p_c = np.array([base_metrics.loc[f"p_c_l0"], base_metrics.loc[f"p_c_l1"]]).squeeze()
    mi_x_Ph = (np.array([mi_x_Ph_l0, mi_x_Ph_l1]) * p_c).sum()
    mi_x_I_Ph = (np.array([mi_x_I_Ph_l0, mi_x_I_Ph_l1]) * p_c).sum()
    
    cont_midict = dict(cont_mi = mi_x_Ph)
    cont_midf = pd.DataFrame([cont_midict]).T
    cont_midf.columns = ["value"]
    return cont_midf

def format_metrics(eval_dict, prefix):
    split_metrics = compute_split_metrics(eval_dict[f"{prefix}_eval_samples"], prefix)
    #combined_metrics = eval_dict[f"{prefix}_eval"].loc[["P_mi","I_P_mi", "P_acc_correct", "I_P_acc_correct"]]
    #combined_metrics.index = [f"{prefix}_{x}" for x in combined_metrics.index]
    combined_metrics = compute_combined_metrics(eval_dict[f"{prefix}_eval"], prefix)
    all_metrics = pd.DataFrame(
        pd.concat((split_metrics, combined_metrics), axis=0)).reset_index()
    all_metrics.columns = ["metric", "value"]
    all_metrics.set_index("metric", inplace=True)
    #all_metrics["origin"] = prefix
    cont_mi = compute_containment(all_metrics)
    all_metrics = pd.concat((all_metrics, cont_mi), axis=0)
    all_metrics.reset_index(inplace=True)
    final_metrics = all_metrics[
        all_metrics["index"].isin([
            "base_mi","P_mi", "I_P_mi","I_P_fth_mi", 
            "reconstructed","encapsulation","perc_P_mi","perc_I_P_mi",
            "perc_encapsulation", "perc_reconstructed", "cont_mi"
        ])]
    final_metrics["metric"] = [METRIC_MAPPING[x] for x in final_metrics["index"]]
    final_metrics.drop("index", axis=1, inplace=True)
    if prefix == "nucgen":
        final_metrics["nucleus"] = True
    elif prefix == "gen":
        final_metrics["nucleus"] = False
    else:
        raise ValueError("Other prefixes no longer supported")
    return final_metrics

def format_sample_eval(eval_dict):
    #run_info = eval_dict["acc_df"].loc[0, ["concept", "model", "k"]].to_dict()
    #test_metrics = format_metrics(eval_dict, "test")
    if eval_dict["gen_eval"] is not None and eval_dict["nucgen_eval"] is not None:
        gen_metrics = format_metrics(eval_dict, "gen")
        nucgen_metrics = format_metrics(eval_dict, "nucgen")
        all_metrics = pd.concat(
            [gen_metrics, nucgen_metrics], axis=0
        ).reset_index(drop=True)
    else:
        raise ValueError("This eval no longer works for test samples")
    
    all_metrics["model"] = eval_dict["model_name"]
    all_metrics["concept"] = eval_dict["concept"]
    all_metrics["k"] = eval_dict["k"]
    all_metrics["maj_acc_test"] = eval_dict["maj_acc_test"]
    return all_metrics

def create_agg_dfs(pairs):
    dfs = []
    for model_name, concept in pairs:
        eval_dir = os.path.join(RESULTS, f"corr_eval/{concept}/{model_name}")
        run_evals = [x for x in os.listdir(eval_dir) if x.endswith(".pkl")]
        for run_eval in run_evals:
            run_eval_path = os.path.join(eval_dir, run_eval)
            with open(run_eval_path, 'rb') as f:      
                eval_dict = pickle.load(f)
            df = format_sample_eval(eval_dict)
            df["run"] = run_eval_path
            dfs.append(df)
            #fth_dfs.append(run_eval["fth_df"])
            #er_dfs.append(run_eval["er_df"])
    return dfs

#%%#################
# Main             #
####################
if __name__=="__main__":
    #args = get_args()
    #logging.info(args)

    agg_pairs = [
        ("gpt2-large", "number"),
        #("bert-base-uncased", "number"),
        ("gpt2-base-french", "gender"),
        #("camembert-base", "gender"),
    ]
    all_dfs = create_agg_dfs(agg_pairs)

    outdir = RESULTS
    all_df = pd.concat(all_dfs,axis=0)
    #all_fth_df = pd.concat(all_fth_dfs,axis=0)
    #all_er_df = pd.concat(all_er_dfs,axis=0)
    all_df.to_csv(os.path.join(outdir, f"corr_res.csv"), index=False)
    #all_fth_df.to_csv(os.path.join(outdir, f"fth.csv"), index=False)
    #all_er_df.to_csv(os.path.join(outdir, f"er.csv"), index=False)
    logging.info("Finished exporting all results.")

