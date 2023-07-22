#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

os.environ['CURL_CA_BUNDLE'] = ''

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
sys.path.append('./src/')

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
from data.create_p_x import load_p_x
from analysis.format_res import get_best_runs
from data.filter_generations import sample_filtered_hs, load_filtered_hs_wff


coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%% Dataset filtering by label helpers
"""
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
        l0_hs, l1_hs = sample_filtered_hs(l0_hs, l1_hs, nsamples)
    return l0_hs, l1_hs
"""
def filter_hs_w_ys(X, facts, foils, y, value):
    idx = np.nonzero(y==value)
    sub_hs, sub_facts, sub_foils = X[idx], facts[idx], foils[idx]
    sub_hs_wff = [x for x in zip(sub_hs, sub_facts, sub_foils)]
    return sub_hs_wff

#%% Result format helpers
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
    for prefix in ["P", "I_P", "baseline"]:
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
    #assert model_name in GPT2_LIST, "Doesn't work for masked anymore"
    run = load_run_output(run_path)
    P, I_P = load_run_Ps(run_path)

    # test set version of the eval
    V, l0_tl, l1_tl = load_model_eval(model_name, concept)
    l0_hs_wff = filter_hs_w_ys(
        run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
    )
    l1_hs_wff = filter_hs_w_ys(
        run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
    )
    if nsamples is not None:
        l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)

    test_eval_samples, test_eval = compute_eval_filtered_hs(
        model_name, concept, P, I_P, l0_hs_wff, l1_hs_wff
    )
    #test_kl_baseline = compute_kl_baseline(
    #    run["X_test"], V, l0_tl, l1_tl, nsamples=nsamples
    #)

    # generated hs version of the eval
    if model_name in GPT2_LIST:
        gen_p_x = load_p_x(model_name, False)
        gen_p_x = None
        gen_l0_hs_wff, gen_l1_hs_wff = load_filtered_hs_wff(
            model_name, nsamples=nsamples
        )
        gen_eval_samples, gen_eval = compute_eval_filtered_hs(
            model_name, concept, P, I_P, gen_l0_hs_wff, gen_l1_hs_wff, gen_p_x
        )
        #gen_Xs = np.vstack([x[0] for x in gen_l0_hs_wff + gen_l1_hs_wff])
        #gen_kl_baseline = compute_kl_baseline(
        #    gen_Xs, V, l0_tl, l1_tl, nsamples=nsamples
        #)

        nuc_p_x = load_p_x(model_name, True)
        nucgen_l0_hs_wff, nucgen_l1_hs_wff = load_filtered_hs_wff(
            model_name, nucleus=True, nsamples=nsamples
        )
        nucgen_eval_samples, nucgen_eval = compute_eval_filtered_hs(
            model_name, concept, P, I_P, nucgen_l0_hs_wff, nucgen_l1_hs_wff, nuc_p_x
        )
        #nucgen_Xs = np.vstack([x[0] for x in nucgen_l0_hs_wff + nucgen_l1_hs_wff])
        #nucgen_kl_baseline = compute_kl_baseline(
        #    nucgen_Xs, V, l0_tl, l1_tl, nsamples=nsamples
        #)
    else:
        gen_eval = None
        gen_eval_samples = None
        nucgen_eval = None
        nucgen_eval_samples = None
        #gen_kl_baseline = None

    #acc_df = create_acc_df(
    #    test_eval, gen_eval, run["diag_eval"], run["usage_eval"], 
    #    run["maj_acc_test"], concept, model_name, run_name, run["config"]["k"]
    #)
    #fth_df = create_fth_df(
    #    test_eval, gen_eval, test_kl_baseline, gen_kl_baseline,
    #    concept, model_name, run_name, run["config"]["k"]
    #)
    #er_df = create_er_df(
    #    test_eval, gen_eval, concept, model_name, run_name, run["config"]["k"]
    #)
    output = dict(
        model_name=model_name,
        concept=concept,
        k=run["config"]["k"],
        maj_acc_test=run["maj_acc_test"],
        test_eval=test_eval,
        test_eval_samples=test_eval_samples,
        gen_eval=gen_eval,
        gen_eval_samples=gen_eval_samples,
        nucgen_eval=nucgen_eval,
        nucgen_eval_samples=nucgen_eval_samples,
        #test_kl_baseline=test_kl_baseline,
        #gen_kl_baseline=gen_kl_baseline,
        #acc_df=acc_df,
        #fth_df=fth_df,
        #er_df=er_df
    )
    return output

#%% LOADERS AND PARAMS    
def compute_eval_pair(model_name, concept, run_output_folder, nsamples):
    rundir = os.path.join(OUT, f"run_output/{concept}/{model_name}/{run_output_folder}")
    if run_output_folder == "230718":
        outdir = os.path.join(RESULTS, f"new_{concept}/{model_name}")
    else:
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
        choices=["230627", "230627_fix", "230628", "230718"],
        help="Run export folder to load"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    pairs = [(args.model, args.concept, args.folder)]
    #pairs = [("gpt2-large", "number", "230628")]
    nsamples = 200
    #nsamples = 10

    logging.info(
        f"Computing run eval from raw run output for"
        f"{args.model} from {args.folder}"
    )

    compute_evals(pairs, nsamples)
    logging.info("Finished exporting all results.")
