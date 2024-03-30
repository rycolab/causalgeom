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
from evals.mi_distributor_mt import MultiTokenDistributor
from evals.mt_mi_computer import compute_mis

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
        "-k",
        type=int,
        default=1,
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
    #argparser.add_argument(
    #    "-htype",
    #    type=str,
    #    choices=["l0_cxt_qxhs_par", "l1_cxt_qxhs_par", 
    #             "l0_cxt_qxhs_bot", "l1_cxt_qxhs_bot", 
    #             "l0_cxt_pxhs", "l1_cxt_pxhs"],
    #    help="Type of test set contexts to compute eval distrib for"
    #)
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept = args.concept
    nucleus = args.nucleus
    k = args.k
    nsamples=args.nsamples
    msamples=args.msamples
    nwords = None
    output_folder = args.out_folder
    run_path = args.run_path
    nruns = 3
    #htype=args.htype
    #model_name = "gpt2-large"
    #concept = "food"
    #nucleus = True
    #k=1
    #nsamples=3
    #msamples=3
    #nwords = 10
    #output_folder = "test"
    #nruns = 1
    #run_path="out/run_output/food/gpt2-large/leace28032024/run_leace_food_gpt2-large_2024-03-28-13:44:28_0_3.pkl"
    #htype = "l1_cxt_qxhs_par"
    

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        
        evaluator = MultiTokenDistributor(
            model_name, 
            concept, 
            nucleus, # nucleus 
            nsamples, #nsamples
            msamples, #msamples
            nwords, #nwords
            run_path, #run_path
            output_folder,
            i
        )
        run_eval_output = evaluator.compute_all_pxs()
        torch.cuda.empty_cache()
        logging.info(f"Eval iteration {i} distributions computed")

        compute_mis(
            model_name, concept, run_path, nucleus, output_folder, i
        )
        logging.info(f"Eval iteration {i} MIs computed")
    logging.info("Finished running all iterations.")
