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
from evals.mi_distributor_utils import get_eval_directory, get_run_path_info
from evals.mi_computer_utils import compute_all_MIs
from evals.eval_utils import load_run_output

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
class MIComputer:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 eval_source, # whether to use samples generated with nucleus sampling
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 #nsamples, # number of test set samples
                 #msamples, # number of dev set samples for each q computation
                 #nwords, # number of words to use from token lists -- GET RID OF THIS
                ):

        run = load_run_output(run_path)
        self.proj_source = run["proj_source"]

        outdir = get_eval_directory(
            "mt_eval", run_path, concept, model_name, self.proj_source,
            output_folder_name, eval_source, iteration
        )
        self.h_distribs_dir = os.path.join(
            outdir, "h_distribs"
        )
        assert os.path.exists(self.h_distribs_dir), "H distribs dir doesn't exist"

        
    #########################################
    # Data handling                         #
    #########################################
    def load_htype_probs(self, htype):
        """ Loads word probabilities exported by MultiTokenDistributor
        pxhs output dim: (n_hs, l0_nwords), (n_hs, l1_nwords), (n_hs, other_nwords)
        qxhs output dim: (n_hs, msamples, l0_nwords), (n_hs, msamples, l1_nwords), (n_hs, msamples, other_nwords)
        """
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

        if htype == "p_x_mid_h":
            l0_stacked_probs = np.vstack(l0_probs)
            l1_stacked_probs = np.vstack(l1_probs)
            other_stacked_probs = np.vstack(other_probs)
        elif htype in ["q_x_mid_hpar", "q_x_mid_hbot"]:
            l0_stacked_probs = np.stack(l0_probs)
            l1_stacked_probs = np.stack(l1_probs)
            other_stacked_probs = np.stack(other_probs)
        else:
            raise ValueError(f"Incorrect htype: {htype}")
        return l0_stacked_probs, l1_stacked_probs, other_stacked_probs
        
    def load_all_pxs(self):
        """ Loads word probabilities exported by MultiTokenDistributor
        p(x | h): (n_hs, l0_nwords), (n_hs, l1_nwords), (n_hs, other_nwords)
        q(x | hpar): (n_hs, msamples, l0_nwords), (n_hs, msamples, l1_nwords), (n_hs, msamples, other_nwords)
        q(x | hbot): (n_hs, msamples, l0_nwords), (n_hs, msamples, l1_nwords), (n_hs, msamples, other_nwords)
        """
        
        p_x_mid_h = self.load_htype_probs("p_x_mid_h")
        q_x_mid_hpar = self.load_htype_probs("q_x_mid_hpar")
        q_x_mid_hbot = self.load_htype_probs("q_x_mid_hbot")

        assert q_x_mid_hpar[0].shape[0] == q_x_mid_hbot[0].shape[0], \
            "Unequal number of samples"
        assert q_x_mid_hpar[0].shape[0] == p_x_mid_h[0].shape[0], \
            "Unequal number of samples"
        
        return p_x_mid_h, q_x_mid_hpar, q_x_mid_hbot

    #########################################
    # Containment and Stability             #
    #########################################
    def compute_run_eval(self):
        all_pxhs, all_qxhpars, all_qxhbots = self.load_all_pxs()

        MIs = compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars)
        return MIs

# %%
def compute_mis(model_name, concept, run_path, 
    eval_source, output_folder, iteration):

    micomputer = MIComputer(
        model_name, 
        concept, 
        eval_source,
        run_path,
        output_folder,
        iteration
    )
    run_eval_output = micomputer.compute_run_eval()
            
    run_metadata = {
        "model_name": model_name,
        "concept": concept,
        "eval_source": eval_source,
        "proj_source": micomputer.proj_source,
        #"nsamples": nsamples,
        #"msamples": msamples,
        "eval_name": output_folder,
        "run_path": run_path,
        "iteration": iteration,
    }
    full_run_output = run_metadata | run_eval_output
    
    # run info
    run_output_dir, run_id = get_run_path_info(run_path)

    # export
    actual_outdir = os.path.join(RESULTS, f"mis/{output_folder}")
    os.makedirs(actual_outdir, exist_ok=True)
    outpath = os.path.join(
        actual_outdir, 
        f"mis_{model_name}_{concept}_{micomputer.proj_source}_"
        f"{run_output_dir}_{output_folder}_run_{run_id}_"
        f"{eval_source}_evaliter_{iteration}.pkl"
    )
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

    #model_name = args.model
    #concept = args.concept
    #source = args.source
    #output_folder = args.out_folder
    #run_path = args.run_path
    #nruns = 3
    
    model_name = "gpt2-large"
    concept = "number"
    source = "test_concept"
    
    output_folder = "june2"
    nruns=1
    run_path=os.path.join(
        OUT, "run_output/june2/number/gpt2-large/gen_ancestral_all/run_leace_number_gpt2-large_gen_ancestral_all_2024-06-02-15:54:03_0_3.pkl"
    )
    

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        compute_mis(
            model_name, concept, run_path, source, 
            output_folder, i
        )
    logging.info("Finished exporting all results.")
