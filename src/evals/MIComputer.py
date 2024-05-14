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
    get_nucleus_arg, get_mt_eval_directory
#from evals.eval_utils import renormalize
from evals.mi_computer_utils import combine_lemma_contexts, \
    compute_all_MIs

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
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 #nsamples, # number of test set samples
                 #msamples, # number of dev set samples for each q computation
                 #nwords, # number of words to use from token lists -- GET RID OF THIS
                ):

        outdir = get_mt_eval_directory(run_path, concept, model_name, 
            output_folder_name, source, iteration)
        self.h_distribs_dir = os.path.join(
            outdir, "h_distribs"
        )
        assert os.path.exists(self.h_distribs_dir), "H distribs dir doesn't exist"

        nucleus = get_nucleus_arg(source)
        #self.nsamples = nsamples
        #self.msamples = msamples
        #self.nwords = nwords
        #_, _, _, _ = prep_generated_data(
        #    model_name, concept, nucleus
        #)
        
    #########################################
    # Data handling                         #
    #########################################
    def load_htype_probs(self, htype):
        htype_dir = os.path.join(self.h_distribs_dir, htype)
        htype_files = os.listdir(htype_dir)
        l0_probs, l1_probs, other_probs = [], [], []
        for fname in tqdm(htype_files):
            fpath = os.path.join(htype_dir, fname)
            with open(fpath, 'rb') as f:      
                sample_l0_probs, sample_l1_probs, sample_other_probs = pickle.load(f)
            l0_probs.append(sample_l0_probs)
            l1_probs.append(sample_l1_probs)
            other_probs.append(sample_other_probs)

        if htype in ["l0_cxt_pxhs", "l1_cxt_pxhs"]:
            # shape: (nsamples, n_l0_words, ), (nsamples, n_l1_words, ), (nsamples, n_other_words, )
            l0_stacked_probs = np.vstack(l0_probs)
            l1_stacked_probs = np.vstack(l1_probs)
            other_stacked_probs = np.vstack(other_probs)
        elif htype in ["l0_cxt_qxhs_par", "l1_cxt_qxhs_par", 
                        "l0_cxt_qxhs_bot", "l1_cxt_qxhs_bot"]:
            # shape: (nsamples, msamlpes, n_l0_words, ), (nsamples, msamlpes, n_l1_words, ), (nsamples, msamlpes, n_other_words, )
            l0_stacked_probs = np.stack(l0_probs)
            l1_stacked_probs = np.stack(l1_probs)
            other_stacked_probs = np.stack(other_probs)
        else:
            raise ValueError(f"Incorrect htype: {htype}")
        return l0_stacked_probs, l1_stacked_probs, other_stacked_probs
        
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
    def compute_run_eval(self):
        l0_qxhpars, l1_qxhpars, l0_qxhbots, l1_qxhbots, l0_pxhs, l1_pxhs = self.load_all_pxs()

        all_pxhs = combine_lemma_contexts(l0_pxhs, l1_pxhs)
        all_qxhpars = combine_lemma_contexts(l0_qxhpars, l1_qxhpars)
        all_qxhbots = combine_lemma_contexts(l0_qxhbots, l1_qxhbots)

        MIs = compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars)
        return MIs

# %%
def compute_mis(model_name, concept, run_path, 
    source, output_folder, iteration):

    micomputer = MIComputer(
        model_name, 
        concept, 
        source,
        run_path,
        output_folder,
        iteration
    )
    run_eval_output = micomputer.compute_run_eval()
            
    run_metadata = {
        "model_name": model_name,
        "concept": concept,
        "source": source,
        #"nsamples": nsamples,
        #"msamples": msamples,
        "eval_name": output_folder,
        "run_path": run_path,
        "iteration": iteration,
    }
    full_run_output = run_metadata | run_eval_output
    
    
    # run info
    rundir = os.path.dirname(run_path)
    rundir_name = os.path.basename(rundir)
    run_id = run_path[-27:-4]

    # export
    actual_outdir = os.path.join(RESULTS, f"mis/{output_folder}")
    os.makedirs(actual_outdir, exist_ok=True)
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
