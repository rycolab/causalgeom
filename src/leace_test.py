#%%
import warnings
import logging
import os
import sys
import argparse
import coloredlogs

import numpy as np
import pickle
import torch
from tqdm import tqdm, trange
import random
import pandas as pd
import time
from datetime import datetime

import torch
import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import SGDClassifier
#from sklearn.decomposition import PCA
#import wandb

#from algorithms.rlace.rlace import solve_adv_game, init_classifier, get_majority_acc, solve_adv_game_param_free

#import algorithms.inlp.debias
#from classifiers.classifiers import BinaryParamFreeClf
#from classifiers.compute_marginals import compute_concept_marginal, compute_pair_marginals
#from utils.cuda_loaders import get_device
#from utils.config_args import get_train_probes_config
#from evals.kl_eval import load_model_eval, compute_eval_filtered_hs
#from evals.usage_eval import full_usage_eval, full_diag_eval
from utils.dataset_loaders import load_processed_data
from data.embed_wordlists.embedder import load_concept_token_lists


from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
concept = "number"
model_name = "gpt2-large"
train_obs = 20000
val_obs = 10000
test_obs = 30000
train_share = 2/6
val_share = 1/6


l0_tl, l1_tl = load_concept_token_lists(concept, model_name)
X, U, y, facts, foils = load_processed_data(concept, model_name)

#%%
from train_probes import get_data_indices

idx_train, idx_val, idx_test = get_data_indices(
    X.shape[0], concept, train_obs, val_obs, test_obs,
        train_share, val_share)
    
X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
U_train, U_val, U_test = U[idx_train], U[idx_val], U[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
facts_train, facts_val, facts_test = facts[idx_train], facts[idx_val], facts[idx_test]
foils_train, foils_val, foils_test = foils[idx_train], foils[idx_val], foils[idx_test]

#%%
from concept_erasure import LeaceEraser

def compute_leace_affine(X_train, y_train):
    X_torch = torch.from_numpy(X_train)
    y_torch = torch.from_numpy(y_train)

    eraser = LeaceEraser.fit(X_torch, y_torch)
    P = (eraser.proj_left @ eraser.proj_right).numpy().T
    I_P = np.eye(X_train.shape[1]) - P.T
    bias = eraser.bias.numpy().T
    return P, I_P, bias
# %%
#P, I_P = compute_leace_Ps(X_train, y_train)
X_test_transf = eraser(torch.from_numpy(X_test)).numpy()

delta = X_test - eraser.bias.numpy()
X_test_check = X_test - (delta @ eraser.proj_right.T.numpy()) @ eraser.proj_left.T.numpy()


#%%
check2 = X_test @ I_P.T + eraser.bias.numpy().T @ P.T
# %%
checker = (X_test_transf - check2)


#%%
X_par = X_test @ P.T - eraser.bias.numpy().T @ P.T

#%%
h = X_test[0]

hbot = h.T @ I_P.T + eraser.bias.numpy().T @ P.T
hpar = h.T @ P.T - eraser.bias.numpy().T @ P.T

# %%
from concept_erasure import LeaceEraser

def compute_leace_affine(X_train, y_train):
    X_torch = torch.from_numpy(X_train)
    y_torch = torch.from_numpy(y_train)

    eraser = LeaceEraser.fit(X_torch, y_torch)
    P = (eraser.proj_left @ eraser.proj_right).numpy()
    I_P = np.eye(X_train.shape[1]) - P
    bias = eraser.bias.numpy()
    return P.T, I_P.T, bias.T

P, I_P, bias = compute_leace_affine(X_train, y_train)

#%%
hbot2 = h.T @ I_P + bias @ P
hpar2 = h.T @ P - bias @ P

# %%
