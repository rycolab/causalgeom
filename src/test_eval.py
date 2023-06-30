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


coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% Helpers
def create_df(records, column_names, concept, model_name, run_name, k):
    df = pd.DataFrame.from_records(
        records, columns=column_names
    )
    #df["run_name"] = run_name
    df["concept"] = concept
    df["model"] = model_name
    df["k"] = k
    #df = df[["concept", "model", "run_name", "k"] + column_names]
    df = df[["concept", "model", "k"] + column_names]
    return df 

#acc
def create_acc_df(test_eval, gen_eval, run_diag_eval, run_usage_eval, 
    maj_acc_test, concept, model_name, run_name, k):
    acc_res = []
    for prefix in ["base", "P", "I_P"]:
        if prefix == "base":
            other_prefix = "original"
        else:
            other_prefix = prefix + "_burn"
        if gen_eval is not None:
            acc_res.append((f"{prefix}", 
                maj_acc_test,
                gen_eval[f"{prefix}_acc_correct"], 
                gen_eval[f"{prefix}_acc_fact_highest"], 
                gen_eval[f"{prefix}_acc_foil_highest"], 
                gen_eval[f"{prefix}_acc_fact_highest_concept"], 
                gen_eval[f"{prefix}_acc_foil_highest_concept"],
                test_eval[f"{prefix}_acc_correct"], 
                test_eval[f"{prefix}_acc_fact_highest"], 
                test_eval[f"{prefix}_acc_foil_highest"], 
                test_eval[f"{prefix}_acc_fact_highest_concept"], 
                test_eval[f"{prefix}_acc_foil_highest_concept"],
                run_diag_eval[f"diag_acc_{other_prefix}_test"],
                run_usage_eval[f"lm_acc_{other_prefix}_test"]
            ))
        else:
            acc_res.append((f"{prefix}", 
                maj_acc_test,
                None,
                None,
                None,
                None,
                None,
                test_eval[f"{prefix}_acc_correct"], 
                test_eval[f"{prefix}_acc_fact_highest"], 
                test_eval[f"{prefix}_acc_foil_highest"], 
                test_eval[f"{prefix}_acc_fact_highest_concept"], 
                test_eval[f"{prefix}_acc_foil_highest_concept"],
                run_diag_eval[f"diag_acc_{other_prefix}_test"],
                run_usage_eval[f"lm_acc_{other_prefix}_test"]
            ))
    column_names = ["prefix", "maj_acc_test",
        "gen_accuracy", "gen_fact_highest", "gen_foil_highest", 
        "gen_fact_highest_concept", "gen_foil_highest_concept",
        "test_accuracy", "test_fact_highest", "test_foil_highest", 
        "test_fact_highest_concept", "test_foil_highest_concept",
        "diag_clf_acc", "curated_data_lm_acc"]
    acc_df = create_df(
        acc_res, column_names, concept, model_name, run_name, k
    )
    return acc_df 

# fth

def create_fth_df(test_eval, gen_eval, test_baseline, gen_baseline,
    concept, model_name, run_name, k):
    fth_res = []
    for prefix in ["I_P", "baseline"]: #"P", 
        if gen_eval is not None and prefix != "baseline":
            fth_res.append((f"{prefix}", 
                #gen_eval[f"{prefix}_faith_kl_all_split"], 
                #gen_eval[f"{prefix}_faith_kl_tgt_split_unnorm"],
                #gen_eval[f"{prefix}_faith_kl_other_unnorm"], 
                gen_eval[f"{prefix}_faith_kl_all_merged"],
                #gen_eval[f"{prefix}_faith_kl_tgt_split"],
                #gen_eval[f"{prefix}_faith_kl_tgt_merged"],
                #gen_eval[f"{prefix}_faith_kl_other"], 
                #test_eval[f"{prefix}_faith_kl_all_split"], 
                #test_eval[f"{prefix}_faith_kl_tgt_split_unnorm"],
                #test_eval[f"{prefix}_faith_kl_other_unnorm"], 
                test_eval[f"{prefix}_faith_kl_all_merged"],
                #test_eval[f"{prefix}_faith_kl_tgt_split"],
                #test_eval[f"{prefix}_faith_kl_tgt_merged"],
                #test_eval[f"{prefix}_faith_kl_other"], 
            ))
        elif gen_eval is None and prefix != "baseline":
            fth_res.append((f"{prefix}", 
                #None,
                #None,
                #None,
                None,
                #None,
                #None,
                #None,
                #test_eval[f"{prefix}_faith_kl_all_split"], 
                #test_eval[f"{prefix}_faith_kl_tgt_split_unnorm"],
                #test_eval[f"{prefix}_faith_kl_other_unnorm"], 
                test_eval[f"{prefix}_faith_kl_all_merged"],
                #test_eval[f"{prefix}_faith_kl_tgt_split"],
                #test_eval[f"{prefix}_faith_kl_tgt_merged"],
                #test_eval[f"{prefix}_faith_kl_other"], 
            ))
        elif prefix == "baseline" and gen_baseline is None:
            fth_res.append((f"{prefix}", 
                #None,
                #None,
                #None,
                None,
                #None,
                #None,
                #None,
                #test_baseline["all_split"],
                #test_baseline["tgt_split_unnorm"],
                #test_baseline["other_unnorm"],
                test_baseline["all_merged"],
                #test_baseline["tgt_split"],
                #test_baseline["tgt_merged"],
                #test_baseline["other"],
            ))
        else:
            fth_res.append((f"{prefix}", 
                #gen_baseline["all_split"],
                #gen_baseline["tgt_split_unnorm"],
                #gen_baseline["other_unnorm"],
                gen_baseline["all_merged"],
                #gen_baseline["tgt_split"],
                #gen_baseline["tgt_merged"],
                #gen_baseline["other"],
                #test_baseline["all_split"],
                #test_baseline["tgt_split_unnorm"],
                #test_baseline["other_unnorm"],
                test_baseline["all_merged"],
                #test_baseline["tgt_split"],
                #test_baseline["tgt_merged"],
                #test_baseline["other"],
            ))
    #column_names = ["prefix", "gen_kl_all_split", "gen_kl_tgt_split_unnorm", 
    #    "gen_kl_other_unnorm", "gen_kl_all_merged", 
    #    "gen_kl_tgt_split", "gen_kl_tgt_merged", "gen_kl_other",
    #    "test_kl_all_split", "test_kl_tgt_split_unnorm", 
    #    "test_kl_other_unnorm", "test_kl_all_merged", 
    #    "test_kl_tgt_split", "test_kl_tgt_merged", "test_kl_other"]
    column_names = ["prefix", "gen_kl_all_merged", "test_kl_all_merged"]
    fth_df = create_df(fth_res, column_names, concept, model_name, run_name, k)
    return fth_df

#er 
def create_er_df(test_eval, gen_eval, concept, model_name, run_name, k):
    er_res = []
    for prefix in ["base", "P", "I_P"]:
        if gen_eval is not None:
            er_res.append((f"{prefix}", 
                gen_eval[f"h_c"], 
                gen_eval[f"{prefix}_h_c_h"],
                gen_eval[f"{prefix}_mi"], 
                test_eval[f"h_c"], 
                test_eval[f"{prefix}_h_c_h"],
                test_eval[f"{prefix}_mi"], 
            ))
        else:
            er_res.append((f"{prefix}", 
                None,
                None,
                None,
                test_eval[f"h_c"], 
                test_eval[f"{prefix}_h_c_h"],
                test_eval[f"{prefix}_mi"], 
            ))
    column_names = ["prefix", "gen_h_c", "gen_h_c_h", "gen_mi",
        "test_h_c", "test_h_c_h", "test_mi"]
    er_df = create_df(er_res, column_names, concept, model_name, run_name, k)
    return er_df

#%% main runner
def filter_test_hs_wff(X, facts, foils, l0_tl, l1_tl, nsamples=None):
    l0_hs = []
    l1_hs = []
    for h, fact, foil in zip(X, facts, foils):
        if fact in l0_tl:
            l0_hs.append((h, fact, foil))
        elif fact in l1_tl:
            l1_hs.append((h, fact, foil))
        else:
            continue

    if nsamples is not None:
        random.shuffle(l0_hs)
        random.shuffle(l1_hs)
        ratio = len(l1_hs)/len(l0_hs)
        if ratio > 1:
            l0_hs = l0_hs[:nsamples]
            l1_hs = l1_hs[:int((nsamples*ratio))]
        else:
            ratio = len(l0_hs) / len(l1_hs)
            l0_hs = l0_hs[:int((nsamples*ratio))]
            l1_hs = l1_hs[:nsamples]
    return l0_hs, l1_hs


def compute_kl_baseline(hs, V, l0_tl, l1_tl, nsamples=200):
    hs_sub = hs[:nsamples*2, :]

    kls = []
    if nsamples*2 <= hs_sub.shape[0]:
        rangeint = nsamples
    else:
        rangeint = int(hs_sub.shape[0] / 2)
    for i in trange(rangeint):
        h1 = hs_sub[i, :]
        h2 = hs_sub[rangeint + i, :]
        h1_base_distribs = get_distribs(h1, V, l0_tl, l1_tl)
        h2_base_distribs = get_distribs(h2, V, l0_tl, l1_tl)
        res = dict(
            all_split=compute_kl(h1_base_distribs["all_split"], h2_base_distribs["all_split"]),
            tgt_split_unnorm=compute_kl(h1_base_distribs["lemma_split"], h2_base_distribs["lemma_split"]),
            tgt_merged_unnorm=compute_kl(h1_base_distribs["lemma_merged"], h2_base_distribs["lemma_merged"]),
            other_unnorm=compute_kl(h1_base_distribs["other"], h2_base_distribs["other"]),
            all_merged=compute_kl(h1_base_distribs["all_merged"], h2_base_distribs["all_merged"]),
            tgt_split=compute_kl(renormalize(h1_base_distribs["lemma_split"]), renormalize(h2_base_distribs["lemma_split"])),
            tgt_merged=compute_kl(renormalize(h1_base_distribs["lemma_merged"]), renormalize(h2_base_distribs["lemma_merged"])),
            other=compute_kl(renormalize(h1_base_distribs["other"]), renormalize(h2_base_distribs["other"])),
        )
        kls.append(res)
    return pd.DataFrame(kls).describe().loc["mean", :]


def compute_run_eval(model_name, concept, run_name, run_path, nsamples=200):
    
    run = load_run_output(run_path)
    P, I_P = load_run_Ps(run_path)

    # test set version of the eval
    V, l0_tl, l1_tl, _ = load_model_eval(model_name, concept)
    #l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
    test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
        run["X_test"], run["facts_test"], run["foils_test"], 
        l0_tl, l1_tl, nsamples=nsamples
    )
    test_eval = compute_eval_filtered_hs(
        model_name, concept, P, I_P, test_l0_hs_wff, test_l1_hs_wff
    )
    test_kl_baseline = compute_kl_baseline(
        run["X_test"], V, l0_tl, l1_tl, nsamples=nsamples
    )

    # generated hs version of the eval
    if model_name in GPT2_LIST:
        gen_l0_hs_wff, gen_l1_hs_wff = load_filtered_hs_wff(
            model_name, nsamples=nsamples
        )
        gen_eval = compute_eval_filtered_hs(
            model_name, concept, P, I_P, gen_l0_hs_wff, gen_l1_hs_wff
        )
        gen_Xs = np.vstack([x[0] for x in gen_l0_hs_wff + gen_l1_hs_wff])
        gen_kl_baseline = compute_kl_baseline(
            gen_Xs, V, l0_tl, l1_tl, nsamples=nsamples
        )
    else:
        gen_eval = None
        gen_kl_baseline = None

    acc_df = create_acc_df(
        test_eval, gen_eval, run["diag_eval"], run["usage_eval"], 
        run["maj_acc_test"], concept, model_name, run_name, run["config"]["k"]
    )
    fth_df = create_fth_df(
        test_eval, gen_eval, test_kl_baseline, gen_kl_baseline,
        concept, model_name, run_name, run["config"]["k"]
    )
    er_df = create_er_df(
        test_eval, gen_eval, concept, model_name, run_name, run["config"]["k"]
    )
    output = dict(
        test_eval=test_eval,
        gen_eval=gen_eval,
        test_kl_baseline=test_kl_baseline,
        gen_kl_baseline=gen_kl_baseline,
        acc_df=acc_df,
        fth_df=fth_df,
        er_df=er_df
    )
    return output

#%% LOADERS AND PARAMS    
def compute_eval_pair(model_name, concept, run_output_folder, nsamples):
    rundir = os.path.join(OUT, f"run_output/{concept}/{model_name}/{run_output_folder}")
    outdir = os.path.join(RESULTS, f"{concept}/{model_name}")
    #outdir = RESULTS
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    runs = [x for x in os.listdir(rundir) if x.endswith(".pkl")]

    for run in runs:
        run_path = os.path.join(rundir, run)
        outpath = os.path.join(outdir, f"eval_{run}")

        if os.path.exists(outpath):
            logging.info(f"Run already evaluated: {run}")
            continue
        else:
            run_eval_output = compute_run_eval(
                model_name, concept, run, run_path, nsamples
            )
            with open(outpath, "wb") as f:
                pickle.dump(run_eval_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Run eval exported: {run}")

    logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}")

def compute_evals(pairs, nsamples):
    for model_name, concept, run_output_folder in pairs:
        compute_eval_pair(model_name, concept, run_output_folder, nsamples)
    logging.info("Finished computing all pairs of evals")

def create_agg_dfs(pairs):
    acc_dfs = []
    fth_dfs = []
    er_dfs = []
    for model_name, concept, _ in pairs:
        eval_dir = os.path.join(RESULTS, f"{concept}/{model_name}")
        run_evals = [x for x in os.listdir(eval_dir) if x.endswith(".pkl")]

        for run_eval in run_evals:
            run_eval_path = os.path.join(eval_dir, run_eval)
            with open(run_eval_path, 'rb') as f:      
                run_eval = pickle.load(f)
            acc_dfs.append(run_eval["acc_df"])
            fth_dfs.append(run_eval["fth_df"])
            er_dfs.append(run_eval["er_df"])
    return acc_dfs, fth_dfs, er_dfs

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
        "-folder",
        type=str,
        choices=["230627", "230627_fix", "230628"],
        help="Run export folder to load"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    #pairs = [
    #    ("gpt2-large", "number", "230627"),
    #    ("bert-base-uncased", "number", "230627"),
    #    ("gpt2-base-french", "gender", "230627"),
    #    ("camembert-base", "gender", "230627"),
    #    ("gpt2-large", "number", "230627_fix"),
    #    ("bert-base-uncased", "number", "230627_fix"),
    #    ("gpt2-base-french", "gender", "230627_fix"),
    #   ("camembert-base", "gender", "230627_fix"),
    #]
    pairs = [(args.model, args.concept, args.folder)]
    nsamples = 200

    logging.info(
        f"Computing run eval from raw run output for"
        f"{args.model} from {args.folder}"
    )

    compute_evals(pairs, nsamples)
    """
    agg_pairs = [
        ("gpt2-large", "number", "230627"),
        ("bert-base-uncased", "number", "230627"),
        ("gpt2-base-french", "gender", "230627"),
        ("camembert-base", "gender", "230627"),
    ]
    all_acc_dfs, all_fth_dfs, all_er_dfs = create_agg_dfs(agg_pairs)

    outdir = RESULTS
    all_acc_df = pd.concat(all_acc_dfs,axis=0)
    all_fth_df = pd.concat(all_fth_dfs,axis=0)
    all_er_df = pd.concat(all_er_dfs,axis=0)
    all_acc_df.to_csv(os.path.join(outdir, f"acc.csv"), index=False)
    all_fth_df.to_csv(os.path.join(outdir, f"fth.csv"), index=False)
    all_er_df.to_csv(os.path.join(outdir, f"er.csv"), index=False)
    """
    logging.info("Finished exporting all results.")

