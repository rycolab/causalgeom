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

from scipy.special import softmax, kl_div

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from evals.kl_eval import load_hs, load_model_eval, load_run_output,\
    get_distribs, normalize_pairs, compute_overall_mi, compute_kl, renormalize, \
        sample_hs

#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# PCA              #
####################
from sklearn.decomposition import PCA

data = np.random.rand(30000,768)
X_pca = PCA(768)
data_pca = X_pca.fit_transform(data)


vec = np.random.rand(768)

h_pca = X_pca.transform(vec.reshape(1,-1)).reshape(-1)
P = np.eye(768, 768)
t1 = X_pca.inverse_transform(P @ h_pca)
t2 = X_pca.inverse_transform((P @ h_pca).reshape(1,-1))

data_pca @ P

