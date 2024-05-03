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
import time
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
from torch.utils.data import DataLoader, Dataset
from abc import ABC
from itertools import zip_longest
from scipy.special import softmax
from scipy.stats import entropy
import math

sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


from evals.mi_distributor_utils import prep_generated_data, compute_inner_loop_qxhs
from utils.lm_loaders import SUPPORTED_AR_MODELS
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
from evals.MIComputer import MIComputer

model_name="gpt2-large"
concept="ambiance"
source="natural"
#nsamples=10
#msamples=15
#nwords=10
#batch_size=64
run_path=os.path.join(
    OUT, "run_output/ambiance/gpt2-large/leace05042024/run_leace_ambiance_gpt2-large_2024-04-05-15:30:40_0_3.pkl"
)
output_folder = "new_mt_eval"
iteration = 0


#%%
micomputer = MIComputer(
    model_name, 
    concept, 
    source,
    run_path, 
    output_folder,
    iteration
)
MIs = micomputer.compute_run_eval()

#%%







