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
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept = args.concept
    source = args.source
    nsamples=args.nsamples
    msamples=args.msamples
    nwords = None
    output_folder = args.out_folder
    run_path = args.run_path
    batch_size = args.batch_size
    n_other_words = args.n_other_words
    nruns = 3
    
    #if source in ['gen_nucleus' 'gen_normal']:
    #    batch_size = 3
    #else:
    #    batch_size = 64
    
    #model_name = "gpt2-large"
    #concept = "food"
    #source = "gen_normal"
    #nsamples=3
    #msamples=3
    #nwords=None
    #output_folder = "test"
    #run_path="out/run_output/number/gpt2-large/leace26032024/run_leace_number_gpt2-large_2024-03-26-19:55:11_0_3.pkl"
    #batch_size = 64
    #nruns = 1
    
    

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        
        evaluator = MultiTokenDistributor(
            model_name, 
            concept, 
            source, # source 
            nsamples, #nsamples
            msamples, #msamples
            nwords, #nwords
            n_other_words, 
            run_path, #run_path
            output_folder,
            i,
            batch_size
        )
        run_eval_output = evaluator.compute_all_pxs()
        torch.cuda.empty_cache()
        logging.info(f"Eval iteration {i} distributions computed")

        compute_mis(
            model_name, concept, run_path, source, output_folder, i
        )
        logging.info(f"Eval iteration {i} MIs computed")
    logging.info("Finished running all iterations.")
