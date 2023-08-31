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
import pandas as pd
from tqdm import tqdm, trange
import pickle
from scipy.special import softmax
from random import sample, shuffle
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
#from data.embed_wordlists.embedder import load_concept_token_lists
from evals.kl_eval import load_run_Ps, load_run_output, load_model_eval,\
    correct_flag, highest_rank, highest_concept
from final_eval import prep_data, compute_inner_loop_qxhs

from data.filter_generations import load_filtered_hs_wff
from evals.run_eval import filter_hs_w_ys, sample_filtered_hs

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
def filter_all_hs_w_ys(X, facts, foils, y):
    l0_hs_wff = filter_hs_w_ys(
        X, facts, foils, y, 0
    )
    l1_hs_wff = filter_hs_w_ys(
        X, facts, foils, y, 1
    )
    return l0_hs_wff, l1_hs_wff

def prep_data_from_test_only(X, facts, foils, y, train_share=0.5):
    l0_hs_wff, l1_hs_wff = filter_all_hs_w_ys(X, facts, foils, y)

    l0_nobs = len(l0_hs_wff)
    train_l0_hs, test_l0_hs = l0_hs_wff[:int(l0_nobs*train_share)], l0_hs_wff[int(l0_nobs*train_share):]

    l1_nobs = len(l1_hs_wff)
    train_l1_hs, test_l1_hs = l1_hs_wff[:int(l1_nobs*train_share)], l1_hs_wff[int(l1_nobs*train_share):]
    return train_l0_hs, test_l0_hs, train_l1_hs, test_l1_hs

def get_hs_proj(hs, P):
    hs_proj = []
    for h, _, _ in hs:
    #h,_,_ = hs_wff[0]
        hs_proj.append(h.T @ P)
    hs_proj = np.vstack(hs_proj)
    #hs_proj_mean = np.mean(hs_proj, axis=0)
    return hs_proj#, hs_proj_mean

def compute_avg_int(h_erase, hs_proj, nsamples, V, processor=None):
    all_probs = []
    idx = np.random.choice(hs_proj.shape[0], nsamples, replace=False)
    for h_proj in hs_proj[idx]:
        #h_sg = hs_proj[0]
        probs = compute_pxh((h_erase + h_proj), V, processor)
        all_probs.append(probs)
    return np.mean(np.vstack(all_probs), axis=0)
 
def score_post_int(base_distrib, I_P_distrib, #l0_avgh_probs, l1_avgh_probs, 
        l0_avgp_probs, l1_avgp_probs, faid, foid, case, l0_tl, l1_tl):
    if case == 0:
        l0id, l1id = faid, foid
    else:
        l0id, l1id = foid, faid
    return dict(
        case=case,
        base_correct = correct_flag(base_distrib[faid], base_distrib[foid]),
        base_correct_highest = highest_rank(base_distrib, faid),
        base_correct_highest_concept = highest_concept(base_distrib, faid, l0_tl, l1_tl),
        I_P_correct = correct_flag(I_P_distrib[faid], I_P_distrib[foid]),
        I_P_l0_highest = highest_rank(I_P_distrib, l0id),
        I_P_l1_highest = highest_rank(I_P_distrib, l1id),
        I_P_l0_highest_concept = highest_concept(I_P_distrib, l0id, l0_tl, l1_tl),
        I_P_l1_highest_concept = highest_concept(I_P_distrib, l1id, l0_tl, l1_tl),
        #avgh_inj0_correct = correct_flag(l0_avgh_probs[l0id], l0_avgh_probs[l1id]),
        #avgh_inj0_l0_highest = highest_rank(l0_avgh_probs, l0id),
        #inj0_l1_highest = highest_rank(l0_avgh_probs, l1id),
        #avgh_inj0_l0_highest_concept = highest_concept(l0_avgh_probs, l0id, l0_tl, l1_tl),
        #inj0_l1_highest_concept = highest_concept(l0_avgh_probs, l1id, l0_tl, l1_tl),
        #avgh_inj1_correct = correct_flag(l1_avgh_probs[l1id], l1_avgh_probs[l0id]),
        #inj1_l0_highest = highest_rank(l1_avgh_probs, l0id),
        #avgh_inj1_l1_highest = highest_rank(l1_avgh_probs, l1id),
        #inj1_l0_highest_concept = highest_concept(l1_avgh_probs, l0id, l0_tl, l1_tl),
        #avgh_inj1_l1_highest_concept = highest_concept(l1_avgh_probs, l1id, l0_tl, l1_tl),
        avgp_inj0_correct = correct_flag(l0_avgp_probs[l0id], l0_avgp_probs[l1id]),
        avgp_inj0_l0_highest = highest_rank(l0_avgp_probs, l0id),
        #inj0_l1_highest = highest_rank(l0_avgp_probs, l1id),
        avgp_inj0_l0_highest_concept = highest_concept(l0_avgp_probs, l0id, l0_tl, l1_tl),
        #inj0_l1_highest_concept = highest_concept(l0_avgp_probs, l1id, l0_tl, l1_tl),
        avgp_inj1_correct = correct_flag(l1_avgp_probs[l1id], l1_avgp_probs[l0id]),
        #inj1_l0_highest = highest_rank(l1_avgp_probs, l0id),
        avgp_inj1_l1_highest = highest_rank(l1_avgp_probs, l1id),
        #inj1_l0_highest_concept = highest_concept(l1_avgp_probs, l0id, l0_tl, l1_tl),
        avgp_inj1_l1_highest_concept = highest_concept(l1_avgp_probs, l1id, l0_tl, l1_tl),
    )

def compute_pxh(h, V, processor=None):
    logits = V @ h
    if processor is not None:
        logits = torch.FloatTensor(logits).unsqueeze(0)
        tokens = torch.LongTensor([0]).unsqueeze(0)
        logits = processor(tokens, logits).squeeze(0).numpy()
    return softmax(logits)

def intervene_test_set(test_hs, case, l0_dev_hs, l1_dev_hs, all_hs, 
    V, l0_tl, l1_tl, P, I_P, nsamples, msamples, nucleus=False):
    #l0_dev_hs_mean = np.mean(l0_dev_hs, axis=0)    
    #l1_dev_hs_mean = np.mean(l1_dev_hs, axis=0)
    #if nucleus:
    #    processor = LogitsProcessorList()
    #    processor.append(TopPLogitsWarper(0.9))
    #else:
    processor=None
    scores = []
    for h, faid, foid in tqdm(test_hs):
        #h, faid, foid = test_hs[0]
        base_distrib = compute_pxh(h, V, processor)
        h_erase = h.T @ I_P
        #I_P_distrib = compute_pxh(h_erase, V, processor)
        inner_qxhs = compute_inner_loop_qxhs(
            "hpar", h, all_hs, P, I_P, V, msamples, processor=processor
        )
        I_P_distrib = inner_qxhs.mean(axis=0)
        #l0_avgh_probs = compute_pxh((h_erase + l0_dev_hs_mean), V, processor)
        #l1_avgh_probs = compute_pxh((h_erase + l1_dev_hs_mean), V, processor)
        l0_avgp_probs = compute_avg_int(h_erase, l0_dev_hs, nsamples, V, processor)
        l1_avgp_probs = compute_avg_int(h_erase, l1_dev_hs, nsamples, V, processor)
        score = score_post_int(
            base_distrib, I_P_distrib, l0_avgp_probs, l1_avgp_probs, 
            faid, foid, case, l0_tl, l1_tl
        )
        scores.append(score)
    return scores

#%%
def compute_int_eval_run(model_name, concept, run, run_path, 
    nsamples_dev=20, nsamples_test=None, msamples=30, 
    nucleus=False, iteration=0):
    P, I_P = load_run_Ps(run_path)
    V, l0_tl, l1_tl = load_model_eval(model_name, concept)

    train_l0_hs, test_l0_hs, train_l1_hs, test_l1_hs = prep_data_from_test_only(
        run["X_test"], run["facts_test"], run["foils_test"], run["y_test"]
    )
    _, _, _, all_hs = prep_data(model_name, nucleus)

    if nsamples_test is not None:
        test_l0_hs, test_l1_hs = sample_filtered_hs(
            test_l0_hs, test_l1_hs, nsamples_test
        )

    l0_train_Phs = get_hs_proj(train_l0_hs, P)
    l1_train_Phs = get_hs_proj(train_l1_hs, P)

    scores_l0 = intervene_test_set(
        test_l0_hs, 0, l0_train_Phs, l1_train_Phs, all_hs, 
        V, l0_tl, l1_tl, P, I_P, nsamples_dev, msamples, nucleus=False
    )
    scores_l1 = intervene_test_set(
        test_l1_hs, 1, l0_train_Phs, l1_train_Phs, all_hs, 
        V, l0_tl, l1_tl, P, I_P, nsamples_dev, msamples, nucleus=False
    )

    scores = pd.DataFrame(scores_l0 + scores_l1)
    scores["model"] = model_name
    scores["concept"] = concept
    scores["nucleus"] = nucleus
    scores["run"] = run_path
    scores["iteration"] = iteration
    scores["dev_total_samples"] = len(train_l0_hs) + len(train_l1_hs)
    scores["dev_nsamples"] = nsamples_dev
    scores["test_total_samples"] = len(test_l0_hs) + len(test_l1_hs)
    scores["test_nsamples"] = nsamples_test
    return scores
    
#%%
def compute_int_eval_folder(model_name, concept, run_output_folder, 
    nsamples_dev, nsamples_test, msamples, nucleus, output_folder, iteration):
    rundir = os.path.join(OUT, f"run_output/{concept}/{model_name}/{run_output_folder}")
    outdir = os.path.join(RESULTS, output_folder)
    #outdir = RESULTS
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    run_files = [x for x in os.listdir(rundir) if x.endswith(".pkl")]

    for run_file in run_files:
        run_path = os.path.join(rundir, run_file)
        outpath = os.path.join(
            outdir, 
            f"{concept}_{model_name}_{nucleus}_{iteration}_intacc_{run_file[:-4]}.csv"
        )

        run = load_run_output(run_path)
        if run["config"]["k"] != 1:
            continue
        elif os.path.exists(outpath):
            logging.info(f"Run already evaluated: {run_path}")
            continue
        else:
            int_accs_df = compute_int_eval_run(
                model_name, concept, run, run_path, 
                nsamples_dev=nsamples_dev, nsamples_test=nsamples_test, 
                msamples=msamples, nucleus=nucleus, iteration=iteration
            )
            int_accs_df.to_csv(outpath)
            #with open(outpath, "wb") as f:
            #    pickle.dump(run_eval_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Run eval exported: {outpath}")

    logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}")


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
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    argparser.add_argument(
        "-out_folder",
        type=str,
        default="test",
        help="Directory for exporting run eval"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    #TODO: train on dev and test on test
    model_name = args.model
    concept = args.concept
    nucleus = args.nucleus
    output_folder = args.out_folder
    nsamples_dev = 20
    nsamples_test = 100
    msamples = 30
    nruns = 3
    #model_name = "gpt2-large"
    #concept = "number"
    #nucleus = True
    #nsamples_dev = 3
    #nsamples_test = 3

    logging.info(
        f"Computing run ints from raw run output for"
        f"{args.model} with nucleus {nucleus}"
    )    

    for folder in ["leacefinal"]:
        for i in range(nruns):
            compute_int_eval_folder(
                model_name, concept, folder, 
                nsamples_dev, nsamples_test, msamples, nucleus,
                output_folder, i
            )
    logging.info("Finished exporting all results.")
