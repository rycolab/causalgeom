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
#from evals.corr_mi_computer_utils import get_corr_mt_eval_directory
from evals.mi_distributor_utils import get_run_path_info
#from evals.mi_computer_utils import compute_all_MIs
from evals.eval_utils import load_run_output
from evals.mi_computer_utils import compute_all_z_distributions, \
    compute_I_C_H
from evals.mt_eval_runner import get_data_type
from evals.MultiTokenDistributor import MultiTokenDistributor

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
class CorrMIComputer:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 nsamples, # number of test set samples
                 #msamples, # number of dev set samples for each q computation
                 #nwords, # number of words to use from token lists -- GET RID OF THIS
                 #n_other_words, # number of other words
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 batch_size, #batch size for the word list probability computation
                 p_new_word=True, # whether to multiply the p(word | h) by p(new_word | h)
                 exist_ok=False, # DEBUG ONLY: export directory exist_ok
                 torch_dtype=torch.float16, # torch data type to use for eval
                ):

        self.model_name = model_name
        self.concept = concept
        self.nsamples = nsamples
        self.msamples = 1 # irrelevant
        self.nwords = None 
        self.n_other_words = 1 # other words not needed for corr I(C;Hbot)
        self.run_path = run_path
        self.output_folder_name = output_folder_name
        self.iteration = iteration
        self.batch_size = batch_size
        self.p_new_word = p_new_word
        self.exist_ok = exist_ok
        self.torch_dtype = torch_dtype

        # Exist ok
        if self.exist_ok:
            logging.warn("DEBUG ONLY: RUNNING WITH EXIST_OK=TRUE")

        # Load run data
        run = load_run_output(self.run_path)
        self.proj_source = run["proj_source"]
        assert run["config"]['model_name'] == model_name, "Run model doesn't match"
        assert run["config"]['concept'] == concept, "Run concept doesn't match"

        
    def get_p_x_mid_hbot(self, eval_source):
        evaluator = MultiTokenDistributor(
            self.model_name, 
            self.concept, 
            eval_source, # eval_source 
            self.nsamples, #nsamples
            self.msamples, #msamples
            self.nwords, #nwords
            self.n_other_words, 
            self.run_path, #run_path
            self.output_folder_name,
            self.iteration,
            self.batch_size,
            p_new_word=self.p_new_word,
            exist_ok=self.exist_ok,
            torch_dtype=self.torch_dtype,
        )
        p_x_mid_hbot_nostack = evaluator.compute_corr_pxhbots()
        p_x_mid_hbot = [np.vstack(x) for x in p_x_mid_hbot_nostack]
        torch.cuda.empty_cache()
        return p_x_mid_hbot
    
    def get_all_p_x_mid_hbots(self):
        train_all_p_x_mid_hbot = self.get_p_x_mid_hbot("train_all")
        train_concept_p_x_mid_hbot = self.get_p_x_mid_hbot("train_concept")
        test_all_p_x_mid_hbot = self.get_p_x_mid_hbot("test_all")
        test_concept_p_x_mid_hbot = self.get_p_x_mid_hbot("test_concept")
        return train_all_p_x_mid_hbot, train_concept_p_x_mid_hbot,\
            test_all_p_x_mid_hbot, test_concept_p_x_mid_hbot

    #########################################
    # Corr MI Computation                   #
    #########################################
    @staticmethod
    def compute_corr_MI(p_x_mid_hbot):
        z_c, z_hbot, _, z_c_mid_hbot, _, _ = compute_all_z_distributions(
            p_x_mid_hbot
        )
        Hz_c, Hz_c_mid_hbot, MIz_c_hbot = compute_I_C_H(
            z_c, z_c_mid_hbot, p_h=z_hbot
        )
        return Hz_c, Hz_c_mid_hbot, MIz_c_hbot

    def compute_all_corr_MIs(self):
        train_all_p_x_mid_hbot, train_concept_p_x_mid_hbot, test_all_p_x_mid_hbot, test_concept_p_x_mid_hbot = self.get_all_p_x_mid_hbots()

        train_all_Hz_C, train_all_Hz_c_mid_hbot, train_all_MIz_c_hbot = self.compute_corr_MI(train_all_p_x_mid_hbot)
        train_concept_Hz_C, train_concept_Hz_c_mid_hbot, train_concept_MIz_c_hbot = self.compute_corr_MI(train_concept_p_x_mid_hbot)
        test_all_Hz_C, test_all_Hz_c_mid_hbot, test_all_MIz_c_hbot = self.compute_corr_MI(test_all_p_x_mid_hbot)
        test_concept_Hz_C, test_concept_Hz_c_mid_hbot, test_concept_MIz_c_hbot = self.compute_corr_MI(test_concept_p_x_mid_hbot)

        output = {
            "train_all_Hz_C": train_all_Hz_C, 
            "train_all_Hz_c_mid_hbot": train_all_Hz_c_mid_hbot, 
            "train_all_MIz_c_hbot": train_all_MIz_c_hbot,
            "train_concept_Hz_C": train_concept_Hz_C, 
            "train_concept_Hz_c_mid_hbot": train_concept_Hz_c_mid_hbot, 
            "train_concept_MIz_c_hbot": train_concept_MIz_c_hbot,
            "test_all_Hz_C": test_all_Hz_C, 
            "test_all_Hz_c_mid_hbot": test_all_Hz_c_mid_hbot, 
            "test_all_MIz_c_hbot": test_all_MIz_c_hbot,
            "test_concept_Hz_C": test_concept_Hz_C, 
            "test_concept_Hz_c_mid_hbot": test_concept_Hz_c_mid_hbot, 
            "test_concept_MIz_c_hbot": test_concept_MIz_c_hbot,
        }
        return output

    def compute_run_corr_eval(self):
        corr_eval_output = self.compute_all_corr_MIs()
        run_metadata = {
            "model_name": self.model_name,
            "concept": self.concept,
            "eval_source": None,
            "eval_name": self.output_folder_name,
            "run_path": self.run_path,
            "iteration": self.iteration,
        }
        full_run_output = run_metadata | corr_eval_output
        return full_run_output


# %%
def compute_corr_mis(
    model_name, concept, nsamples, 
    run_path, output_folder_name, iteration, 
    batch_size, p_new_word=True, exist_ok=False, 
    torch_dtype=torch.float16):

    corrmicomputer = CorrMIComputer(
        model_name, concept, nsamples, 
        run_path, output_folder_name, iteration, 
        batch_size, 
        p_new_word=p_new_word, exist_ok=exist_ok, 
        torch_dtype=torch_dtype
    )
    run_corr_eval_output = corrmicomputer.compute_run_corr_eval()

    logging.info(f"Corr MIs {iteration} output:\n {run_corr_eval_output}")
    # export
    run_output_dir, run_id = get_run_path_info(run_path)

    actual_outdir = os.path.join(RESULTS, f"corr_mis/{output_folder_name}")
    os.makedirs(actual_outdir, exist_ok=True)
    outpath = os.path.join(
        actual_outdir, 
        f"corr_mis_{model_name}_{concept}_{corrmicomputer.proj_source}_"
        f"{run_output_dir}_{output_folder_name}_run_{run_id}_"
        f"evaliter_{iteration}.pkl"
    )
    with open(outpath, "wb") as f:
        pickle.dump(run_corr_eval_output, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Run corr eval exported: {outpath}")
    

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
        "-eval_source",
        type=str,
        choices=["train_all", "train_concept", "test_all", "test_concept", 
                 "gen_ancestral_concept", "gen_nucleus_concept", 
                 "gen_ancestral_all", "gen_nucleus_all"],
        help=("Which samples to use for eval."
             "train: train set samples from LEACE train/val/test data"
             "test: test set samples from LEACE train/val/test data"
             "gen_ancestral: generated by model using ancestral sampling"
             "gen_nucleus: generated by model using ancestral sampling"
             "_concept: only include contexts s.t. next token is supposed to have concept value != n/a"
             "_all: no filtering of contexts")
    )
    argparser.add_argument(
        "-nsamples",
        type=int,
        help="Number of samples for outer loops"
    )
    #argparser.add_argument(
    #    "-msamples",
    #    type=int,
    #    help="Number of samples for inner loops"
    #)
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
    argparser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="Batch size for word probability computation"
    )
    #argparser.add_argument(
    #    "-n_other_words",
    #    type=int,
    #    default=500,
    #    help="Number of other words to compute probabilities for"
    #)
    argparser.add_argument(
        "-torch_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        help="Data type to cast all tensors to during evaluation",
        default="float16"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    #model_name = args.model
    #concept = args.concept
    #nsamples= args.nsamples
    #run_path = args.run_path
    #output_folder_name = args.out_folder
    #batch_size = args.batch_size
    #nruns = 3
    #p_new_word = True
    #exist_ok = False
    #torch_dtype = get_data_type(args.torch_dtype)

    model_name = "gpt2-large"
    concept = "number"
    nsamples= 5
    run_path = os.path.join(
        OUT,
        "run_output/june2/number/gpt2-large/gen_ancestral_all/run_leace_number_gpt2-large_gen_ancestral_all_2024-06-02-15:54:03_0_3.pkl"
    )
    output_folder_name = "corr_test"
    batch_size = 64
    nruns=1
    p_new_word=True
    exist_ok=False
    torch_dtype=torch.float16

    for i in range(nruns):
        logging.info(f"Computing corr eval number {i}")
        compute_corr_mis(
            model_name, concept, nsamples, 
            run_path, output_folder_name, i, 
            batch_size, p_new_word=p_new_word, exist_ok=exist_ok, 
            torch_dtype=torch_dtype
        )
    logging.info("Finished exporting all corr eval results.")
