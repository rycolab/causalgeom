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
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.eval_utils import load_run_Ps, load_run_output, load_model_eval,\
    renormalize
from data.filter_generations import load_generated_hs_wff
from data.data_utils import filter_hs_w_ys, sample_filtered_hs

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% COMPUTING H(X; H_parallel | C)
def compute_inner_loop_qxhs(mode, h, all_hs, P, I_P, V, msamples, processor=None):
    """ mode param determines whether averaging over hbot or hpar"""
    all_pxnewh = []
    idx = np.arange(0, all_hs.shape[0])
    np.random.shuffle(idx)
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
    return np.vstack(all_pxnewh)

def compute_qxhs(c_hs, all_hs, inner_mode, I_P, P, V, msamples, nucleus=False):
    c_qxhs = []
    if nucleus:
        processor = LogitsProcessorList()
        processor.append(TopPLogitsWarper(0.9))
    else:
        processor=None
    for h,_,_ in tqdm(c_hs):
        inner_qxhs = compute_inner_loop_qxhs(
            inner_mode, h, all_hs, P, I_P, V, msamples, processor=processor
        )
        c_qxhs.append(inner_qxhs.mean(axis=0))
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
def compute_pxhs(c_hs, V, nucleus=False):
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
def prep_generated_data(model_name, nucleus):
    l0_hs_wff, l1_hs_wff, other_hs = load_generated_hs_wff(
        model_name, load_other=True, nucleus=nucleus
    )
    all_concept_hs = [x for x,_,_ in l0_hs_wff + l1_hs_wff]
    other_hs_no_x = [x for x,_ in other_hs]
    all_hs = np.vstack(all_concept_hs + other_hs_no_x)

    p_c = compute_p_c_bin(l0_hs_wff, l1_hs_wff)
    return p_c, l0_hs_wff, l1_hs_wff, all_hs


def compute_all_pxs(l0_hs_wff, l1_hs_wff, all_hs, I_P, P, V, 
    nsamples, msamples, nucleus):
    
    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_qxhs_par = compute_qxhs(
        l0_hs_n, all_hs, "hbot", I_P, P, V, msamples, nucleus=nucleus
    )
    l1_qxhs_par = compute_qxhs(
        l1_hs_n, all_hs, "hbot", I_P, P, V, msamples, nucleus=nucleus
    )

    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_qxhs_bot = compute_qxhs(
        l0_hs_n, all_hs, "hpar", I_P, P, V, msamples, nucleus=nucleus
    )
    l1_qxhs_bot = compute_qxhs(
        l1_hs_n, all_hs, "hpar", I_P, P, V, msamples, nucleus=nucleus
    )

    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_pxhs = compute_pxhs(l0_hs_n, V, nucleus=nucleus)
    l1_pxhs = compute_pxhs(l1_hs_n, V, nucleus=nucleus)
    return l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs


def compute_containment(l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs, 
    l0_tl, l1_tl, p_c):
    # H(X|H_par, C)
    cont_l0_ent_qxhcs = compute_avg_of_cond_ents(l0_qxhs_par, 0, l0_tl, l1_tl)
    cont_l1_ent_qxhcs = compute_avg_of_cond_ents(l1_qxhs_par, 1, l0_tl, l1_tl)
    cont_ent_qxcs = (p_c * np.array([cont_l0_ent_qxhcs, cont_l1_ent_qxhcs])).sum()

    # H(X|C)
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
        ent_pxc=ent_pxc,
        cont_l0_mi=cont_l0_mi,
        cont_l1_mi=cont_l1_mi,
        cont_mi=cont_mi
    )

#def compute_h_x_c(p_x, p_c, l0_tl, l1_tl):
#    l0_p_x = renormalize(p_x[l0_tl])
#    l1_p_x = renormalize(p_x[l1_tl])
#    h_l0_p_x = entropy(l0_p_x)
#    h_l1_p_x = entropy(l1_p_x)
#    return (p_c * np.array([h_l0_p_x, h_l1_p_x])).sum()

def compute_stability(l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs, 
    l0_tl, l1_tl, p_c):
    # H(X | C)
    #stab_ent_x_c = compute_h_x_c(p_x, p_c, l0_tl, l1_tl)

    #H(X | H,C)
    stab_ent_xhc_l0 = compute_avg_of_cond_ents(l0_pxhs, 0, l0_tl, l1_tl)
    stab_ent_xhc_l1 = compute_avg_of_cond_ents(l1_pxhs, 1, l0_tl, l1_tl)
    stab_ent_xhc = (p_c * np.array([stab_ent_xhc_l0, stab_ent_xhc_l1])).sum()

    #H(X| H_bot,C)
    stab_l0_ent_qxhcs = compute_avg_of_cond_ents(l0_qxhs_bot, 0, l0_tl, l1_tl)
    stab_l1_ent_qxhcs = compute_avg_of_cond_ents(l1_qxhs_bot, 1, l0_tl, l1_tl)
    stab_ent_qxcs = (p_c * np.array([stab_l0_ent_qxhcs, stab_l1_ent_qxhcs])).sum()

    stab_l0_mi = stab_l0_ent_qxhcs - stab_ent_xhc_l0
    stab_l1_mi = stab_l1_ent_qxhcs - stab_ent_xhc_l1
    stab_mi = stab_ent_qxcs - stab_ent_xhc

    logging.info(f"Stability metrics: {stab_l0_mi}, {stab_l1_mi}, {stab_mi}")
    return dict(
        stab_l0_ent_qxhcs=stab_l0_ent_qxhcs,
        stab_l1_ent_qxhcs=stab_l1_ent_qxhcs,
        stab_ent_qxcs=stab_ent_qxcs,
        stab_ent_xhc_l0=stab_ent_xhc_l0,
        stab_ent_xhc_l1=stab_ent_xhc_l1,
        stab_ent_xhc=stab_ent_xhc,
        stab_l0_mi=stab_l0_mi,
        stab_l1_mi=stab_l1_mi,
        stab_mi=stab_mi
    )

#%% COMPUTE CONCEPT MIs
def compute_p_c_bin(l0_hs, l1_hs):
    c_counts = np.array([len(l0_hs), len(l1_hs)])
    p_c = c_counts / np.sum(c_counts)
    return p_c

def compute_bin_pch(pxh, l0_tl, l1_tl):
    pch_l0 = pxh[l0_tl].sum()
    pch_l1 = pxh[l1_tl].sum()
    pch_other = np.delete(pxh, np.hstack((l0_tl, l1_tl))).sum()
    pch = renormalize(np.array([pch_l0, pch_l1, pch_other]))
    pch_bin = renormalize(np.array([pch_l0, pch_l1]))
    return pch_bin

def compute_pchs_from_pxhs(pxhs, l0_tl, l1_tl):
    pchs = []
    for pxh in pxhs:
        #pxh = inner_pxhs[0]
        pch_bin = compute_bin_pch(pxh, l0_tl, l1_tl)
        pchs.append(pch_bin)
    return np.vstack(pchs)

def compute_qchs(c_hs, all_hs, inner_mode, I_P, P, V, l0_tl, l1_tl, 
    msamples, nucleus):
    qchs = []
    if nucleus:
        processor = LogitsProcessorList()
        processor.append(TopPLogitsWarper(0.9))
    else:
        processor=None
    for h,_,_ in tqdm(c_hs):
        #h,_,_ = c_hs[0]
        inner_qxhs = compute_inner_loop_qxhs(
            inner_mode, h, all_hs, P, I_P, V, msamples, processor=processor
        )
        qch = compute_pchs_from_pxhs(inner_qxhs, l0_tl, l1_tl).mean(axis=0)
        qchs.append(qch)
    return np.vstack(qchs)

def compute_concept_mis(l0_hs_wff, l1_hs_wff, all_hs, I_P, P, 
    V, l0_tl, l1_tl, nsamples, msamples, p_c, nucleus):
    # H(C)
    ent_pc = entropy(p_c)

    # H(C | H_bot)
    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_qchs_bot = compute_qchs(
        l0_hs_n, all_hs, "hpar", I_P, P, V, l0_tl, l1_tl, msamples, nucleus
    )
    l1_qchs_bot = compute_qchs(
        l1_hs_n, all_hs, "hpar", I_P, P, V, l0_tl, l1_tl, msamples, nucleus
    )
    qchs_bot = np.vstack([l0_qchs_bot, l1_qchs_bot])
    ent_qchs_bot = entropy(qchs_bot, axis=1).mean()

    # H(C | H_par)
    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_qchs_par = compute_qchs(
        l0_hs_n, all_hs, "hbot", I_P, P, V, l0_tl, l1_tl, msamples, nucleus
    )
    l1_qchs_par = compute_qchs(
        l1_hs_n, all_hs, "hbot", I_P, P, V, l0_tl, l1_tl, msamples, nucleus
    )
    qchs_par = np.vstack([l0_qchs_par, l1_qchs_par])
    ent_qchs_par = entropy(qchs_par, axis=1).mean()

    # H(C | H)
    l0_hs_n, l1_hs_n = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)
    l0_pxhs = compute_pxhs(l0_hs_n, V, nucleus)
    l1_pxhs = compute_pxhs(l1_hs_n, V, nucleus)
    l0_pchs = compute_pchs_from_pxhs(l0_pxhs, l0_tl, l1_tl)
    l1_pchs = compute_pchs_from_pxhs(l1_pxhs, l0_tl, l1_tl)
    pchs = np.vstack([l0_pchs, l1_pchs])
    ent_pchs = entropy(pchs, axis=1).mean()

    return dict(
        ent_qchs_bot = ent_qchs_bot,
        ent_qchs_par = ent_qchs_par,
        ent_pchs = ent_pchs,
        ent_pc = ent_pc,
        mi_c_hbot = ent_pc - ent_qchs_bot,
        mi_c_hpar = ent_pc - ent_qchs_par,
        mi_c_h = ent_pc - ent_pchs,
    )

#%%
def compute_res_run(model_name, concept, run, run_path, nsamples, msamples, nucleus):
    #run = load_run_output(run_path)
    P, I_P, _ = load_run_Ps(run_path)

    # test set version of the eval
    V, l0_tl, l1_tl = load_model_eval(model_name, concept)

    #p_x = load_p_x(model_name, nucleus)
    p_c, l0_hs_wff, l1_hs_wff, all_hs = prep_generated_data(model_name, nucleus)

    l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs = compute_all_pxs(
        l0_hs_wff, l1_hs_wff, all_hs, I_P, P, V, nsamples, msamples, nucleus=False
    )
    containment_res = compute_containment(
        l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs, l0_tl, l1_tl, p_c
    )
    stability_res = compute_stability(
        l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs, l0_tl, l1_tl, p_c
    )
    concept_mis = compute_concept_mis(l0_hs_wff, l1_hs_wff, all_hs, I_P, P, 
        V, l0_tl, l1_tl, nsamples, msamples, p_c, nucleus=False)
    return containment_res | stability_res | concept_mis
    

def compute_eval(model_name, concept, run_output_folder, k,
    nsamples, msamples, nucleus, output_folder, iteration):
    rundir = os.path.join(
        OUT, f"run_output/{concept}/{model_name}/{run_output_folder}"
    )
    #if run_output_folder == "230718":
    #    outdir = os.path.join(RESULTS, f"{output_folder}/new_{concept}/{model_name}")
    #else:
    outdir = os.path.join(RESULTS, f"{output_folder}/{concept}/{model_name}")
    #outdir = RESULTS
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    run_files = [x for x in os.listdir(rundir) if x.endswith(".pkl")]
    random.shuffle(run_files)

    for run_file in run_files:
        run_path = os.path.join(rundir, run_file)
        outpath = os.path.join(
            outdir, 
            f"{concept}_{model_name}_nuc_{nucleus}_{iteration}_{run_file[:-4]}.pkl"
        )

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
                "run_path": run_path,
                "iteration": iteration
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
        "-nsamples",
        type=int,
        help="Number of samples for outer loops"
    )
    argparser.add_argument(
        "-msamples",
        type=int,
        help="Number of samples for inner loops"
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

    #model_name = args.model
    #concept = args.concept
    #nucleus = args.nucleus
    #k = args.k
    #nsamples=args.nsamples
    #msamples=args.msamples
    #output_folder = args.out_folder
    #nruns = 3
    model_name = "gpt2-large"
    concept = "number"
    nucleus = True
    k=1
    nsamples=3
    msamples=3
    output_folder = "finaleval_test"
    nruns = 1
    run_output_folders = ["leace_new"]
    

    for folder in run_output_folders:
        for i in range(nruns):
            compute_eval(
                model_name, concept, folder, k, nsamples, msamples, nucleus,
                output_folder, i
            )
    logging.info("Finished exporting all results.")
