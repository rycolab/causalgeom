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
from scipy.stats import entropy
from tqdm import trange

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.kl_eval import load_run_Ps, load_run_output, load_model_eval
from data.filter_generations import load_filtered_hs_wff
from evals.run_eval import filter_hs_w_ys, sample_filtered_hs
from evals.run_int import get_hs_proj

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
model_name = "gpt2-large"
concept = "number"
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-26-23:00:59_0_3.pkl")
#run_path = os.path.join(OUT, "run_output/gender/gpt2-base-french/230627_fix/run_gpt2-base-french_theta_k128_Plr0.01_Pms16,41,61,81,101_clflr0.01_clfms26,51,76_2023-06-28-09:52:58_0_2.pkl")
nsamples = 400

run_path = os.path.join(OUT, "run_output/number/gpt2-large/230627/run_gpt2-large_theta_k1_Plr0.001_Pms31,76_clflr0.0003_clfms31_2023-06-26-23:02:09_0_3.pkl")

run = load_run_output(run_path)
P, I_P = load_run_Ps(run_path)

# test set version of the eval
V, l0_tl, l1_tl = load_model_eval(model_name, concept)

#l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
#test_l0_hs_wff, test_l1_hs_wff = filter_test_hs_wff(
#    run["X_test"], run["facts_test"], run["foils_test"], 
#    l0_tl, l1_tl, nsamples=nsamples
#)
l0_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 0
)
l1_hs_wff = filter_hs_w_ys(
    run["X_test"], run["facts_test"], run["foils_test"], run["y_test"], 1
)
if nsamples is not None:
    l0_hs_wff, l1_hs_wff = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)


#%%
from data.create_p_x import load_p_x
nucleus = False
p_x = load_p_x(model_name, nucleus)
h_x = entropy(p_x)

#%%
from evals.kl_eval import renormalize


#%%

from evals.kl_eval import get_all_distribs, get_distrib_key, renormalize
c_index = 0
h,_,_ = l0_hs_wff[0]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, V, l0_tl, l1_tl
)

key = get_distrib_key(c_index)
pxPh = renormalize(P_distribs[key])

# %%
