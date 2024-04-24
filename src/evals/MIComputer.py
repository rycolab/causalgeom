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
#from scipy.stats import entropy
from tqdm import trange
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch 
from itertools import zip_longest
from scipy.special import softmax
from scipy.stats import entropy

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


from utils.lm_loaders import SUPPORTED_AR_MODELS
from evals.mi_distributor_utils import prep_generated_data, \
    get_nucleus_arg
from evals.eval_utils import renormalize

#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
#from utils.lm_loaders import get_model, get_tokenizer
#from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
class MIComputer:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 source, # whether to use samples generated with nucleus sampling
                 h_distribs_dir, # path containing test sample distributions for eval
                 #nsamples, # number of test set samples
                 #msamples, # number of dev set samples for each q computation
                 #nwords, # number of words to use from token lists -- GET RID OF THIS
                ):

        nucleus = get_nucleus_arg(source)
        #self.nsamples = nsamples
        #self.msamples = msamples
        #self.nwords = nwords
        self.h_distribs_dir = h_distribs_dir
        self.p_c, _, _, _ = prep_generated_data(
            model_name, concept, nucleus
        )

        
    #########################################
    # Data handling                         #
    #########################################
    def load_htype_probs(self, htype):
        htype_dir = os.path.join(self.h_distribs_dir, htype)
        htype_files = os.listdir(htype_dir)
        l0_probs, l1_probs = [], []
        for fname in tqdm(htype_files):
            fpath = os.path.join(htype_dir, fname)
            with open(fpath, 'rb') as f:      
                sample_l0_probs, sample_l1_probs = pickle.load(f)
            l0_probs.append(sample_l0_probs)
            l1_probs.append(sample_l1_probs)

        l0_stacked_probs = np.vstack(l0_probs)
        l1_stacked_probs = np.vstack(l1_probs)
        return l0_stacked_probs, l1_stacked_probs
        
    def load_all_pxs(self):
        l0_cxt_qxhs_par = self.load_htype_probs("l0_cxt_qxhs_par")
        l1_cxt_qxhs_par = self.load_htype_probs("l1_cxt_qxhs_par")
        l0_cxt_qxhs_bot = self.load_htype_probs("l0_cxt_qxhs_bot")
        l1_cxt_qxhs_bot = self.load_htype_probs("l1_cxt_qxhs_bot")
        l0_cxt_pxhs = self.load_htype_probs("l0_cxt_pxhs")
        l1_cxt_pxhs = self.load_htype_probs("l1_cxt_pxhs")

        assert l0_cxt_qxhs_par[0].shape[0] == l0_cxt_qxhs_bot[0].shape[0], \
            "Unequal number of samples"
        assert l1_cxt_qxhs_par[0].shape[0] == l1_cxt_qxhs_bot[0].shape[0], \
            "Unequal number of samples"
        assert l0_cxt_qxhs_par[0].shape[0] == l0_cxt_pxhs[0].shape[0], \
            "Unequal number of samples"
        assert l1_cxt_qxhs_par[0].shape[0] == l1_cxt_pxhs[0].shape[0], \
            "Unequal number of samples"
        
        return l0_cxt_qxhs_par, l1_cxt_qxhs_par, l0_cxt_qxhs_bot, \
            l1_cxt_qxhs_bot, l0_cxt_pxhs, l1_cxt_pxhs

    #########################################
    # Containment and Stability             #
    #########################################
    @staticmethod
    def compute_avg_of_cond_ents(pxs):
        ents = []
        #assert case in [0,1], "Incorrect case"
        #case_pxs = pxs[case]
        for p in pxs:
            p_x_c = renormalize(p)
            ents.append(entropy(p_x_c))
        return np.mean(ents)

    @staticmethod
    def compute_ent_of_avg(pxs):
        #assert case in [0,1], "Incorrect case"
        #case_pxs = pxs[case]
        mean_px = pxs.mean(axis=0)
        p_x_c = renormalize(mean_px)
        return entropy(p_x_c)

    def compute_containment(self, l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs):
        # H(X|H_par, C)
        cont_l0_ent_qxhcs = self.compute_avg_of_cond_ents(l0_qxhs_par[0])
        cont_l1_ent_qxhcs = self.compute_avg_of_cond_ents(l1_qxhs_par[1])
        cont_ent_qxcs = (self.p_c * np.array([cont_l0_ent_qxhcs, cont_l1_ent_qxhcs])).sum()

        # H(X|C)
        l0_ent_pxc = self.compute_ent_of_avg(l0_pxhs[0])
        l1_ent_pxc = self.compute_ent_of_avg(l1_pxhs[1])
        ent_pxc = (self.p_c * np.array([l0_ent_pxc, l1_ent_pxc])).sum()

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

    def compute_stability(self, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs):
        #H(X | H,C)
        stab_ent_xhc_l0 = self.compute_avg_of_cond_ents(l0_pxhs[0])
        stab_ent_xhc_l1 = self.compute_avg_of_cond_ents(l1_pxhs[1])
        stab_ent_xhc = (self.p_c * np.array([stab_ent_xhc_l0, stab_ent_xhc_l1])).sum()

        #H(X| H_bot,C)
        stab_l0_ent_qxhcs = self.compute_avg_of_cond_ents(l0_qxhs_bot[0])
        stab_l1_ent_qxhcs = self.compute_avg_of_cond_ents(l1_qxhs_bot[1])
        stab_ent_qxcs = (self.p_c * np.array([stab_l0_ent_qxhcs, stab_l1_ent_qxhcs])).sum()

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

    #########################################
    # Erasure and Encapsulation             #
    #########################################
    @staticmethod
    def compute_pchs_from_pxhs(pxhs):
        """ pxhs: tuple (p(l0_words | h), p(l1_words | h))
        for n h's, i.e., element 0 is a n x n_l0_words 
        dimensional np.array
        """
        pchs = []
        for pxh in zip(*pxhs):
            pch_l0 = pxh[0].sum()
            pch_l1 = pxh[1].sum()
            pch_bin = renormalize(np.array([pch_l0, pch_l1]))
            pchs.append(pch_bin)
        return np.vstack(pchs)

    def compute_concept_mis(self, l0_qxhs_par, l1_qxhs_par, \
        l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs):
        
        # H(C)
        ent_pc = entropy(self.p_c)

        # H(C | H_bot)
        l0_qchs_bot = self.compute_pchs_from_pxhs(l0_qxhs_bot)
        l1_qchs_bot = self.compute_pchs_from_pxhs(l1_qxhs_bot)
        qchs_bot = np.vstack([l0_qchs_bot, l1_qchs_bot])
        ent_qchs_bot = entropy(qchs_bot, axis=1).mean()

        # H(C | H_par)
        l0_qchs_par = self.compute_pchs_from_pxhs(l0_qxhs_par)
        l1_qchs_par = self.compute_pchs_from_pxhs(l1_qxhs_par)
        qchs_par = np.vstack([l0_qchs_par, l1_qchs_par])
        ent_qchs_par = entropy(qchs_par, axis=1).mean()

        # H(C | H)
        l0_pchs = self.compute_pchs_from_pxhs(l0_pxhs)
        l1_pchs = self.compute_pchs_from_pxhs(l1_pxhs)
        pchs = np.vstack([l0_pchs, l1_pchs])
        ent_pchs = entropy(pchs, axis=1).mean()

        res = dict(
            ent_qchs_bot = ent_qchs_bot,
            ent_qchs_par = ent_qchs_par,
            ent_pchs = ent_pchs,
            ent_pc = ent_pc,
            mi_c_hbot = ent_pc - ent_qchs_bot,
            mi_c_hpar = ent_pc - ent_qchs_par,
            mi_c_h = ent_pc - ent_pchs,
        )
        logging.info(f"Concept MI metrics I(C; H): {res['mi_c_h']},"
            f"I(C; Hbot): {res['mi_c_hbot']}, I(C; Hpar): {res['mi_c_hpar']}")
        return res

    #########################################
    # Run complete eval                     #
    #########################################
    def compute_run_eval(self):
        l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, \
            l0_pxhs, l1_pxhs = self.load_all_pxs()
        containment_res = self.compute_containment(
            l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs
        )
        stability_res = self.compute_stability(
            l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs
        )
        concept_mis = self.compute_concept_mis( 
            l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs
        )
        return containment_res | stability_res | concept_mis

# %%
def compute_mis(model_name, concept, run_path, 
    source, output_folder, iteration):
    #rundir = os.path.join(
    #    OUT, f"run_output/{concept}/{model_name}/{run_output_folder}"
    #)

    rundir = os.path.dirname(run_path)
    rundir_name = os.path.basename(rundir)

    run_id = run_path[-27:-4]

    outdir = os.path.join(
        os.path.dirname(rundir), 
        f"mt_eval_{rundir_name}/{output_folder}/run_{run_id}/source_{source}/evaliter_{iteration}"
    )
    h_distribs_dir = os.path.join(
        outdir, "h_distribs"
    )
    assert os.path.exists(h_distribs_dir), "H distribs dir doesn't exist"

    micomputer = MIComputer(
        model_name, 
        concept, 
        source,
        h_distribs_dir, #where the distributions for eval were exported
    )
    run_eval_output = micomputer.compute_run_eval()
            
    run_metadata = {
        "model_name": model_name,
        "concept": concept,
        "source": source,
        #"nsamples": nsamples,
        #"msamples": msamples,
        "run_path": run_path,
        "iteration": iteration
    }
    full_run_output = run_metadata | run_eval_output
    actual_outdir = os.path.join(RESULTS, "mis")
    outpath = os.path.join(actual_outdir, f"mis_{model_name}_{concept}_{rundir_name}_{output_folder}_run_{run_id}_source_{source}_evaliter_{iteration}.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(full_run_output, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Run eval exported: {outpath}")
    

#%%#################
# Main             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Formatting Results Tables')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["number", "gender", "food", "ambiance", "service", "noise"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Models to create embedding files for"
    )
    argparser.add_argument(
        "-source",
        type=str,
        choices=["natural", "gen_nucleus", "gen_normal"],
        help="Which samples to use for eval"
    )
    argparser.add_argument(
        "-run_path",
        type=str,
        default=None,
        help="Run to evaluate"
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

    model_name = args.model
    concept = args.concept
    source = args.source
    output_folder = args.out_folder
    run_path = args.run_path
    nruns = 3
    
    #model_name = "gpt2-large"
    #concept = "food"
    #source = "gen_nucleus"
    #k=1
    #nsamples=3
    #msamples=3
    #nwords = None
    #output_folder = "n100"
    #nruns=3
    #run_path="out/run_output/food/gpt2-large/leace28032024/run_leace_food_gpt2-large_2024-03-28-13:44:28_0_3.pkl"
    

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        compute_mis(
            model_name, concept, run_path, source, 
            output_folder, i
        )
    logging.info("Finished exporting all results.")
