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
import torch
import random 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
from data.embed_wordlists.embedder import load_concept_token_lists
from utils.dataset_loaders import load_processed_data
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#####################
# Creating Nice Graphs #
########################
#fpath = os.path.join(DATASETS, "processed/linzen/ar/linzen_gpt2-large_ar.pkl")
#with open(fpath,"rb") as f:
#    data = pickle.load(f)
# %%
from utils.dataset_loaders import load_processed_data
concept = "number" 
model_name = "gpt2-large"
X, U, y, facts, foils = load_processed_data(concept, model_name)

#%%
idx = np.arange(0, X.shape[0])
np.random.shuffle(idx)
X = X[idx]
facts = facts[idx]
foils = foils[idx]

#%%
l0_tl, l1_tl = load_concept_token_lists(concept, model_name)

# %%




# %%
