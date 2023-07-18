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

from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
from evals.kl_eval import load_model_eval
#from data.embed_wordlists.embedder import load_concept_token_lists
#from evals.kl_eval import load_run_Ps, load_run_output, \
#    compute_eval_filtered_hs, load_model_eval, compute_kl, \
#        renormalize, get_distribs
from data.filter_generations import load_filtered_hs_wff, sample_filtered_hs
from evals.eval_run import filter_hs_w_ys
from evals.kl_eval import get_distribs, correct_flag, highest_rank, highest_concept
from utils.dataset_loaders import load_processed_data

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
concept = "gender"
model = "gpt2-base-french"
X, U, y, facts, foils = load_processed_data(concept, model)
V, l0_tl, l1_tl = load_model_eval(model, concept)
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
l0_df = pd.DataFrame(l0_hs_correct, columns=["h", "fact", "foil"])
l0_df["y"] = 0
l1_df = pd.DataFrame(l1_hs_correct, columns=["h", "fact", "foil"])
l1_df["y"] = 1
X = pd.concat((l0_df, l1_df), axis=0).reset_index(drop=True)

nobs = X.shape[0]
idx = np.arange(0, nobs)
np.random.shuffle(idx)

X_train = X.loc[idx[:int(nobs*.7)], :]
X_test = X.loc[idx[int(nobs*.7):], :]

#%%
# TODO: set up the crossvalidation loop, pick best alpha and report test score
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=5, test_size=.25)
for i, (train_index, test_index) in enumerate(rs.split(X_train)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
    X_train_split = X_train.loc[train_index, :]
    X_val_split = X_train.loc[test_index, :]

    
    
# %%
