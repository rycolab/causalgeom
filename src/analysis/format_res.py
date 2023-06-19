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
from evals.kl_eval import load_run_output, load_run_Ps, get_distribs, \
    normalize_pairs, compute_overall_mi, compute_kl, renormalize,\
        compute_kls_from_run_output
from utils.dataset_loaders import load_hs, load_model_eval
from utils.lm_loaders import BERT_LIST, GPT2_LIST

#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# TABLE MAKING.    #
####################
#with open(run_output, 'rb') as f:      
#    run = pickle.load(f)

#%% Diag eval
def format_acc_res(diag_eval_dict, usage_eval_dict):
    diag_usage_res = dict(
        orig=dict(
            test_diag_loss=diag_eval_dict["diag_loss_original_test"],
            test_diag_acc=diag_eval_dict["diag_acc_original_test"],
            test_lm_loss=usage_eval_dict["lm_loss_original_test"],
            test_lm_acc=usage_eval_dict["lm_acc_original_test"],
        ),
        P=dict(
            test_diag_loss=diag_eval_dict["diag_loss_P_burn_test"],
            test_diag_acc=diag_eval_dict["diag_acc_P_burn_test"],
            test_lm_loss=usage_eval_dict["lm_loss_P_burn_test"],
            test_lm_acc=usage_eval_dict["lm_acc_P_burn_test"],
        ),
        I_P=dict(
            test_diag_loss=diag_eval_dict["diag_loss_I_P_burn_test"],
            test_diag_acc=diag_eval_dict["diag_acc_I_P_burn_test"],
            test_lm_loss=usage_eval_dict["lm_loss_I_P_burn_test"],
            test_lm_acc=usage_eval_dict["lm_acc_I_P_burn_test"],
        ),
    )
    return pd.DataFrame(diag_usage_res).T 

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

#%%#################
# BASELINE         #
####################
def get_baseline_kls(concept, model_name, nsamples=200):
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
    return desc_kls

#%%
def recompute_eval(concept_name, model_name, run_output_path, outdir, nsamples):
    desc_outfile = os.path.join(outdir, f"kl_mi_descs_{model_name}_{concept_name}.csv")
    concept_outfile = os.path.join(outdir, f"kl_mi_concept_{model_name}_{concept_name}.csv")
    other_outfile = os.path.join(outdir, f"kl_mi_other_{model_name}_{concept_name}.csv")

    kl_descs, concept_kls, other_kls = compute_kls_from_run_output(
        concept_name, model_name, run_output_path, nsamples)
    kl_descs.to_csv(desc_outfile)
    concept_kls.to_csv(concept_outfile)
    other_kls.to_csv(other_outfile)
    logging.info(f"Exported KLs for all subsets of hs.")

    return kl_descs, concept_kls, other_kls

#%%#################
# Main             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Formatting Results Tables')
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
    argparser.add_argument(
        "-useRun",
        action="store_true",
        default=False,
        help="Whether to load and use metrics computed at the end of the run",
    )
    argparser.add_argument(
        "-exportAcc",
        action="store_true",
        default=False,
        help="Whether to load and format accuracy metrics from the end of run",
    )
    return argparser.parse_args()

def get_best_runs(model_name, concept_name):
    run_output_dir = os.path.join(OUT, f"run_output/{concept_name}/{model_name}")
    if model_name == "bert-base-uncased" and concept_name == "number":
        return os.path.join(run_output_dir, "230614_test/run_bert-base-uncased_k1_Plr0.003_Pms11,21,31,41,51_clflr0.003_clfms200_2023-06-16-12:07:45_0_1.pkl")
    elif model_name == "gpt2-large" and concept_name == "number":
        return os.path.join(run_output_dir, "230614/run_gpt2-large_k1_Plr0.001_Pms31_clflr0.0003_clfms31_2023-06-16-12:07:54_0_1.pkl")
    elif model_name == "camembert-base" and concept_name == "gender":
        return None
    elif model_name == "gpt2-base-french" and concept_name == "gender":
        return os.path.join(run_output_dir, "230614/run_gpt2-base-french_k1_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-16-12:14:50_0_1.pkl")
    else:
        raise ValueError(f"No best run for combination of {model_name} and {concept_name}")

if __name__ == '__main__':
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept_name = args.concept
    useRun = args.useRun
    #model_name = "gpt2-large"
    #concept_name = "number"
    #useRun=False
    exportAcc=True
    nsamples=1000
    #suffix = "nopca"
    
    OUTDIR = os.path.join(RESULTS, f"{concept_name}/{model_name}")
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    logging.info(f"Formatting and exporting run results for {model_name}, {concept_name}")

    run_output_path = get_best_runs(model_name, concept_name)
    run_output = load_run_output(run_output_path)

    if not useRun:
        logging.info(f"Running KL and MI eval for {model_name}, {concept_name}")
        
        raw_outdir = os.path.join(OUT, "raw_results")
        if not os.path.exists(raw_outdir):
            os.makedirs(raw_outdir)

        #TODO: do this with Xtest from run output once available
        klmidesc, _, _ = recompute_eval(concept_name, model_name, run_output_path, raw_outdir, nsamples)
    else:
        klmidesc = run_output["kl_eval"]

    #klmi_results_path = os.path.join(OUT, f"raw_results/kl_mi_{model_name}_{concept_name}.csv")
    #raw_results = pd.read_csv(raw_results_path, index_col=0)

    if exportAcc:
        diag_usage_res_df = format_acc_res(run_output["diag_eval"], run_output["usage_eval"])
        diag_usage_res_path = os.path.join(OUTDIR, f"diag_usage_res.csv")
        diag_usage_res_df.to_csv(diag_usage_res_path)
        logging.info(f"Exported diag_usage results to: {diag_usage_res_path}")

    P_kls = get_full_kls_df(klmidesc, "P")
    I_P_kls = get_full_kls_df(klmidesc, "I_P")

    P_fth_res_path = os.path.join(OUTDIR, f"fth_res_P.csv")
    P_kls.to_csv(P_fth_res_path, index=False)
    logging.info(f"Exported P_fth results to: {P_fth_res_path}")

    I_P_fth_res_path = os.path.join(OUTDIR, f"fth_res_I_P.csv")
    I_P_kls.to_csv(I_P_fth_res_path, index=False)
    logging.info(f"Exported I_P_fth results to: {I_P_fth_res_path}")

    full_ers = get_full_er_df(klmidesc) 

    er_res_path = os.path.join(OUTDIR, f"er_res.csv")
    full_ers.to_csv(er_res_path, index=False)
    logging.info(f"Exported erasure results to: {er_res_path}")

    baseline_kls = get_baseline_kls(concept_name, model_name)
    baseline_kls_path = os.path.join(OUTDIR, "fth_baseline.csv")
    baseline_kls.to_csv(baseline_kls_path, index=True)
    logging.info(f"Exported baseline KL results to: {baseline_kls_path}")