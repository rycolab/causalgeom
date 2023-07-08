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
from tqdm import trange, tqdm

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
from data.embed_wordlists.embedder import load_concept_token_lists
from evals.kl_eval import load_run_Ps, load_run_output, \
    compute_eval_filtered_hs, load_model_eval, compute_kl, \
        renormalize, get_distribs
from analysis.format_res import get_best_runs
from data.filter_generations import load_filtered_hs, load_filtered_hs_wff
from evals.eval_run import filter_test_hs_wff
from evals.kl_eval import get_distribs, correct_flag, highest_rank, highest_concept
from utils.dataset_loaders import load_processed_data
from evals.eval_run import sample_filtered_hs, filter_hs_w_ys

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
concept = "gender"
model = "gpt2-base-french"
X, U, y, facts, foils = load_processed_data(concept, model)
V, l0_tl, l1_tl, _ = load_model_eval(model, concept)
# %%
#l0_hs, l1_hs = filter_test_hs_wff(X, facts, foils, l0_tl, l1_tl)

l0_hs_wff = filter_hs_w_ys(X, facts, foils, y, 0)
l1_hs_wff = filter_hs_w_ys(X, facts, foils, y, 1)

#nsamples=None
#l0_hs, l1_hs = sample_filtered_hs(l0_hs_wff, l1_hs_wff, nsamples)

# %%
from test_inj import get_base_correct
from operator import itemgetter 

def filter_correct_hs(hs, value, V, l0_tl, l1_tl):
    correct_stats = get_base_correct(hs, value, V, l0_tl, l1_tl)
    df = pd.DataFrame(correct_stats)
    idx = df[df["base_correct_highest_concept"] == 1].index.to_list()
    hs_correct = itemgetter(*idx)(hs)
    return hs_correct

l0_hs_correct = filter_correct_hs(l0_hs_wff, 0, V, l0_tl, l1_tl)
l1_hs_correct = filter_correct_hs(l1_hs_wff, 1, V, l0_tl, l1_tl)

#%%
import random
X = pd.DataFrame(l0_hs_correct + l1_hs_correct)

nobs = len(hs_correct)
idx = np.arange(0, nobs)
np.random.shuffle(idx)

X_train = X.loc[idx[:int(nobs*.7)], :]
X_test = X.loc[idx[int(nobs*.7):], :]

#%%
# TODO: set up the crossvalidation loop, pick best alpha and report test score