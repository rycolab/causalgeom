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
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import trange
from transformers import TopPLogitsWarper, LogitsProcessorList

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.kl_eval import load_run_Ps, load_run_output, load_model_eval, renormalize
from data.filter_generations import load_filtered_hs_wff
from evals.run_eval import filter_hs_w_ys, sample_filtered_hs
from evals.run_int import get_hs_proj
from evals.kl_eval import compute_p_c_bin

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% COMPUTING H(X; H_parallel | C)
def compute_inner_loop_qxhs(mode, h, all_hs, P, I_P, V, msamples, processor=None):
    """ mode param determines whether averaging over hbot or hpar"""
    all_pxnewh = []
    idx = np.random.choice(all_hs.shape[0], msamples+1, replace=False)
    for other_h in all_hs[idx[:msamples]]:
        if mode == "hbot":
            newh = other_h.T @ I_P + h.T @ P
        elif mode == "hpar":
            newh = h.T @ I_P + other_h.T @ P
        else:
            raise ValueError(f"Incorrect mode {mode}")
        logits = V @ newh
        if processor is not None:
            logits = torch.FloatTensor(logits).unsqueeze(0)
            tokens = torch.LongTensor([0]).unsqueeze(0)
            logits = processor(tokens, logits).squeeze(0).numpy()
        pxnewh = softmax(logits)
        all_pxnewh.append(pxnewh)
    all_pxnewh = np.vstack(all_pxnewh).mean(axis=0)
    return all_pxnewh

def compute_concept_qxhs(c_hs, all_hs, inner_mode, I_P, P, V, msamples, nucleus=False):
    c_qxhs = []
    if nucleus:
        processor = LogitsProcessorList()
        processor.append(TopPLogitsWarper(0.9))
    else:
        processor=None
    for h,_,_ in tqdm(c_hs):
        inner_qxh = compute_inner_loop_qxhs(
            inner_mode, h, all_hs, P, I_P, V, msamples, processor=processor
        )
        c_qxhs.append(inner_qxh)
    return np.vstack(c_qxhs)

def compute_avg_of_cond_ents(pxs, case, l0_tl, l1_tl):
    ents = []
    for p in pxs:
    #p = pxhs[0]
        if case == 0:
            p_x_c = renormalize(p[l0_tl])
        elif case == 1:
            p_x_c = renormalize(p[l1_tl])
        else:
            raise ValueError("Incorrect case")
        ents.append(entropy(p_x_c))
    return np.mean(ents)

#%% COMPUTING H(X | C)
def compute_concept_pxhs(c_hs, V, nucleus=False):
    c_pxhs = []
    if nucleus:
        processor = LogitsProcessorList()
        processor.append(TopPLogitsWarper(0.9))
    else:
        processor=None
    for h,_,_ in tqdm(c_hs):
        logits = V @ h
        if processor is not None:
            logits = torch.FloatTensor(logits).unsqueeze(0)
            tokens = torch.LongTensor([0]).unsqueeze(0)
            logits = processor(tokens, logits).squeeze(0).numpy()
        pxh = softmax(logits)
        c_pxhs.append(pxh)
    return np.vstack(c_pxhs)

def compute_ent_of_avg(pxs, case, l0_tl, l1_tl):
    mean_px = pxs.mean(axis=0)
    if case == 0:
        p_x_c = renormalize(mean_px[l0_tl])
    elif case == 1:
        p_x_c = renormalize(mean_px[l1_tl])
    else:
        raise ValueError("Incorrect case")
    return entropy(p_x_c)

#%%
def prep_data(model_name, nsamples):
    l0_hs_wff, l1_hs_wff, other_hs = load_filtered_hs_wff(model_name, load_other=True)
    all_concept_hs = [x for x,_,_ in l0_hs_wff + l1_hs_wff]
    other_hs_no_x = [x for x,_ in other_hs]
    all_hs = np.vstack(all_concept_hs + other_hs_no_x)

    p_c = compute_p_c_bin(l0_hs_wff, l1_hs_wff)
    l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    return p_c, l0_hs_wff, l1_hs_wff, all_hs


def compute_all_pxs(l0_hs_wff, l1_hs_wff, all_hs, I_P, P, V, msamples, nucleus):
    l0_qxhs_par = compute_concept_qxhs(
        l0_hs_wff, all_hs, "hbot", I_P, P, V, msamples, nucleus=nucleus
    )
    l1_qxhs_par = compute_concept_qxhs(
        l1_hs_wff, all_hs, "hbot", I_P, P, V, msamples, nucleus=nucleus
    )

    l0_qxhs_bot = compute_concept_qxhs(
        l0_hs_wff, all_hs, "hpar", I_P, P, V, msamples, nucleus=nucleus
    )
    l1_qxhs_bot = compute_concept_qxhs(
        l1_hs_wff, all_hs, "hpar", I_P, P, V, msamples, nucleus=nucleus
    )

    l0_pxhs = compute_concept_pxhs(l0_hs_wff, V, nucleus=nucleus)
    l1_pxhs = compute_concept_pxhs(l1_hs_wff, V, nucleus=nucleus)
    return l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs


def compute_containment(l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs, 
    l0_tl, l1_tl, p_c):
    # H(X|H_par, C)
    cont_l0_ent_qxhcs = compute_avg_of_cond_ents(l0_qxhs_par, 0, l0_tl, l1_tl)
    cont_l1_ent_qxhcs = compute_avg_of_cond_ents(l1_qxhs_par, 1, l0_tl, l1_tl)
    cont_ent_qxcs = (p_c * np.array([cont_l0_ent_qxhcs, cont_l1_ent_qxhcs])).sum()

    #H(X|C)
    l0_ent_pxc = compute_ent_of_avg(l0_pxhs, 0, l0_tl, l1_tl)
    l1_ent_pxc = compute_ent_of_avg(l1_pxhs, 1, l0_tl, l1_tl)
    ent_pxc = (p_c * np.array([l0_ent_pxc, l1_ent_pxc])).sum()

    cont_l0_mi = l0_ent_pxc - cont_l0_ent_qxhcs
    cont_l1_mi = l1_ent_pxc - cont_l1_ent_qxhcs
    cont_mi = ent_pxc - cont_ent_qxcs

    logging.info(f"Containment metrics: {cont_l0_mi}, {cont_l1_mi}, {cont_mi}")
    return dict(
        cont_l0_ent_qxhcs=cont_l0_ent_qxhcs,
        cont_l1_ent_qxhcs=cont_l1_ent_qxhcs,
        cont_ent_qxcs=cont_ent_qxcs,
        l0_ent_pxc=l0_ent_pxc,
        l1_ent_pxc=l1_ent_pxc,
        cont_l0_mi=cont_l0_mi,
        cont_l1_mi=cont_l1_mi,
        cont_mi=cont_mi
    )


def compute_stability(l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs, 
    l0_tl, l1_tl, p_c):
    #H(X | H,C)
    stab_ent_xhc_l0 = compute_avg_of_cond_ents(l0_pxhs, 0, l0_tl, l1_tl)
    stab_ent_xhc_l1 = compute_avg_of_cond_ents(l1_pxhs, 1, l0_tl, l1_tl)
    stab_ent_xhc = (p_c * np.array([stab_ent_xhc_l0, stab_ent_xhc_l1])).sum()

    #H(X| H_bot,C)
    stab_l0_ent_qxhcs = compute_avg_of_cond_ents(l0_qxhs_bot, 0, l0_tl, l1_tl)
    stab_l1_ent_qxhcs = compute_avg_of_cond_ents(l1_qxhs_bot, 1, l0_tl, l1_tl)
    stab_ent_qxcs = (p_c * np.array([stab_l0_ent_qxhcs, stab_l1_ent_qxhcs])).sum()

    stab_l0_mi = stab_l0_ent_qxhcs - stab_ent_xhc_l0
    stab_l1_mi = stab_l0_ent_qxhcs - stab_ent_xhc_l0
    stab_mi = stab_ent_qxcs - stab_ent_xhc

    logging.info(f"Stability metrics: {stab_l0_mi}, {stab_l1_mi}, {stab_mi}")
    return dict(
        stab_l0_ent_qxhcs=stab_l0_ent_qxhcs,
        stab_l1_ent_qxhcs=stab_l1_ent_qxhcs,
        stab_ent_qxcs=stab_ent_qxcs,
        stab_ent_xhc_l0=stab_ent_xhc_l0,
        stab_ent_xhc_l1=stab_ent_xhc_l1,
        stab_l0_mi=stab_l0_mi,
        stab_l1_mi=stab_l1_mi,
        stab_mi=stab_mi
    )

#%%
def compute_res_run(model_name, concept, run, run_path, nsamples, msamples, nucleus):
    #run = load_run_output(run_path)
    P, I_P = load_run_Ps(run_path)

    # test set version of the eval
    V, l0_tl, l1_tl = load_model_eval(model_name, concept)

    p_c, l0_hs_wff, l1_hs_wff, all_hs = prep_data(model_name, nsamples)
    l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs = compute_all_pxs(
        l0_hs_wff, l1_hs_wff, all_hs, I_P, P, V, msamples, nucleus
    )
    containment_res = compute_containment(
        l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs, l0_tl, l1_tl, p_c
    )
    stability_res = compute_stability(
        l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs, l0_tl, l1_tl, p_c
    )
    return containment_res | stability_res

def compute_eval(model_name, concept, run_output_folder, k, nsamples, msamples, nucleus):
    rundir = os.path.join(OUT, f"run_output/{concept}/{model_name}/{run_output_folder}")
    if run_output_folder == "230718":
        outdir = os.path.join(RESULTS, f"stabcont/new_{concept}/{model_name}")
    else:
        outdir = os.path.join(RESULTS, f"stabcont/{concept}/{model_name}")
    #outdir = RESULTS
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    run_files = [x for x in os.listdir(rundir) if x.endswith(".pkl")]

    for run_file in run_files:
        run_path = os.path.join(rundir, run_file)
        outpath = os.path.join(outdir, f"{concept}_{model_name}_nuc_{nucleus}_{run_file[:-4]}.pkl")

        run = load_run_output(run_path)
        if run["config"]["k"] != k:
            continue
        elif os.path.exists(outpath):
            logging.info(f"Run already evaluated: {run_path}")
            continue
        else:
            run_eval_output = compute_res_run(
                model_name, concept, run, run_path, nsamples, msamples, nucleus
            )
            run_metadata = {
                "model_name": model_name,
                "concept": concept,
                "k": k,
                "nucleus": nucleus,
                "nsamples": nsamples,
                "msamples": msamples,
                "run_path": run_path
            }
            full_run_output = run_metadata | run_eval_output
            with open(outpath, "wb") as f:
                pickle.dump(full_run_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Run eval exported: {run_path}")
    logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}, k:{k}")

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
    #argparser.add_argument(
    #    "-folder",
    #    type=str,
    #    choices=["230627", "230627_fix", "230628", "230718"],
    #    help="Run export folder to load"
    #)
    argparser.add_argument(
        "-k",
        type=int,
        help="K value for the runs"
    )
    argparser.add_argument(
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept = args.concept
    nucleus = args.nucleus
    k = args.k
    #model_name = "gpt2-large"
    #concept = "number"
    #nucleus = False
    #k=1
    nsamples=50
    msamples=10
    #nsamples=3
    #msamples=3

    for folder in ["230627", "230627_fix", "230628", "230718"]:
        compute_eval(
            model_name, concept, folder, k, nsamples, msamples, nucleus
        )
    logging.info("Finished exporting all results.")
