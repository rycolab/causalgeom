#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
#import torch
import time
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
from evals.MultiTokenDistributor import MultiTokenDistributor
from evals.MIComputer import compute_mis

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%##########################
# Command Line Arg Handling #
#############################
def get_data_type(type_str):
    if type_str == "float32":
        return torch.float32
    elif type_str == "float16":
        return torch.float16
    elif type_str == "bfloat16":
        return torch.bfloat16
    else:
        raise NotImplementedError(f"Data type {type_str} not implemented")


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
    argparser.add_argument(
        "-msamples",
        type=int,
        help="Number of samples for inner loops"
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
    argparser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="Batch size for word probability computation"
    )
    argparser.add_argument(
        "-n_other_words",
        type=int,
        default=500,
        help="Number of other words to compute probabilities for"
    )
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

    model_name = args.model
    concept = args.concept
    eval_source = args.eval_source
    nsamples= args.nsamples
    msamples= args.msamples
    nwords = None
    output_folder = args.out_folder
    run_path = args.run_path
    batch_size = args.batch_size
    n_other_words = args.n_other_words
    torch_dtype = get_data_type(args.torch_dtype)
    nruns = 3
    exist_ok=False
    
    #model_name = "llama2"
    #concept = "food"
    #eval_source = "gen_nucleus_concept"
    #nsamples=10
    #msamples=3
    #nwords=None
    #n_other_words=10
    #output_folder = "test"
    #run_path="out/run_output/food/llama2/leace27032024/run_leace_food_llama2_2024-03-27-14:58:45_0_3.pkl"
    #batch_size = 8
    #nruns = 1    
    #exist_ok=True
    #torch_dtype = torch.float16

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        
        evaluator = MultiTokenDistributor(
            model_name, 
            concept, 
            eval_source, # eval_source 
            nsamples, #nsamples
            msamples, #msamples
            nwords, #nwords
            n_other_words, 
            run_path, #run_path
            output_folder,
            i,
            batch_size,
            torch_dtype=torch_dtype,
            exist_ok=exist_ok
        )
        run_eval_output = evaluator.compute_all_pxs()
        torch.cuda.empty_cache()
        logging.info(f"Eval iteration {i} distributions computed")

        compute_mis(
            model_name, concept, run_path, eval_source, output_folder, i
        )
        logging.info(f"Eval iteration {i} MIs computed")
    logging.info("Finished running all iterations.")
