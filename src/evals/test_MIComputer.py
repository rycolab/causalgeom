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


#from evals.mi_distributor_utils import prep_generated_data, compute_inner_loop_qxhs
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
concept="food"
source="natural"
#nsamples=5
#msamples=5
#nwords=5
#n_other_words=10
#batch_size=32
run_path=os.path.join(
    OUT, "run_output/food/gpt2-large/leacefinal/run_leace_food_gpt2-large_2024-05-04-12:04:06_0_3.pkl"
)
output_folder = "may10"
iteration = 0

computer = MIComputer(
    model_name,
    concept,
    source,
    run_path,
    output_folder,
    iteration,
)

#%%
computer.compute_run_eval()
#l0_qxhpars, l1_qxhpars, l0_qxhbots, l1_qxhbots, l0_pxhs, l1_pxhs = distributor.load_all_pxs()


#%%
from evals.mi_computer_utils import combine_lemma_contexts, \
    compute_all_z_distributions, compute_all_q_distributions

all_pxhs = combine_lemma_contexts(l0_pxhs, l1_pxhs)
all_qxhpars = combine_lemma_contexts(l0_qxhpars, l1_qxhpars)
all_qxhbots = combine_lemma_contexts(l0_qxhbots, l1_qxhbots)

z_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c = compute_all_z_distributions(all_pxhs)
qbot_c, qbot_c_mid_hbot, qbot_x_mid_c, qbot_x_mid_hbot_c = compute_all_q_distributions(all_qxhbots)
qpar_c, qpar_c_mid_hpar, qpar_x_mid_c, qpar_x_mid_hpar_c = compute_all_q_distributions(all_qxhpars)

#%%
from evals.mi_computer_utils import compute_H_x_mid_h_c

#def compute_I_X_H_mid_C(p_c, p_x_mid_c, p_c_mid_h, p_x_mid_h_c):
""" I(X ; H | C)
in: 
- p(c): (cdim)
- p(x | c): (cdim x xdim)
- p(c | h): (hdim x c_dim)
- p(x | h, c): (h_dim x c_dim x x_dim)
out: H(X|C), H(X|H, C), I(X;H|C)
"""
p_c = z_c
p_x_mid_c = z_x_mid_c
p_c_mid_h = z_c_mid_h
p_x_mid_h_c = z_x_mid_h_c

# H(X | C)
H_x_c = p_c @ entropy(p_x_mid_c, axis=1)

# H(X | H, C)
H_x_mid_h_c = compute_H_x_mid_h_c(p_x_mid_h_c, p_c_mid_h)

MI_x_h_mid_c = H_x_c - H_x_mid_h_c
#return H_x_c, H_x_mid_h_c, MI_x_h_mid_c
# %%
from evals.mi_computer_utils import compute_all_MIs

compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars)