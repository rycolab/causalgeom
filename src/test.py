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

#from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizerFast, BertForMaskedLM

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset
from abc import ABC

from scipy.special import softmax, kl_div

#sys.path.append('..')
sys.path.append('./src/')

from paths import OUT, HF_CACHE, LINZEN_PREPROCESSED

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "linzen"

if MODEL_NAME == "gpt2":
    DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_ar.pkl"
elif MODEL_NAME == "bert-base-uncased":
    DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_masked.pkl"
else:
    DATASET = None

with open(DATASET, 'rb') as f:      
    data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])
    h = data["h"]
    del data

#%%
if MODEL_NAME == "gpt2":
    TOKENIZER = GPT2TokenizerFast.from_pretrained(
        MODEL_NAME, model_max_length=512
    )
    MODEL = None
elif MODEL_NAME == "bert-base-uncased":
    TOKENIZER = BertTokenizerFast.from_pretrained(
        MODEL_NAME, model_max_length=512
    )
    MODEL = BertForMaskedLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=HF_CACHE, 
        is_decoder=False
    )
else:
    logging.warn("INVALID MODEL")

MASK_TOKEN_ID = TOKENIZER.mask_token_id
word_embeddings = MODEL.bert.embeddings.word_embeddings.weight
bias = MODEL.cls.predictions.decoder.bias
V = torch.cat(
    (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()


#%%
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
else: 
    torch.device("cpu")
    logging.warning("No GPU found")
"""

#%%
WORDLIST_PATH = "../datasets/processed/wordlists/bert-base-uncased_wordlist.csv"
VERBLIST_PATH = "../datasets/processed/linzen_verb_list/linzen_verb_list_final.pkl"

#%%
wl = pd.read_csv(WORDLIST_PATH, index_col=0)
wl.drop_duplicates(inplace=True)

word_tok = wl["input_id"].to_numpy()
word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
#count_sort_ind = np.argsort(-word_tok_counts)
#word_tok_unq[count_sort_ind]

#%%
with open(VERBLIST_PATH, 'rb') as f:      
    vl = pickle.load(f)

#%%
vl["sverb_ntok"] = vl["sverb_tok"].apply(lambda x: len(x))
vl["pverb_ntok"] = vl["pverb_tok"].apply(lambda x: len(x))
vl["1tok"] = (vl["sverb_ntok"] == 1) & (vl["pverb_ntok"]==1)

vl.drop(vl[vl["1tok"] != True].index, inplace=True)

vl_sg_tok = vl["sverb_tok"].apply(lambda x: x[0]).to_numpy()
#vl_sg_tok_unq, vl_sg_tok_counts = np.unique(vl_sg_tok, return_counts=True)
#count_sort_ind = np.argsort(-vl_sg_tok_counts)
#vl_sg_tok_unq[count_sort_ind]
#vl_sg_tok_counts[count_sort_ind]

vl_pl_tok = vl["pverb_tok"].apply(lambda x: x[0]).to_numpy()
#vl_pl_tok_unq, vl_pl_tok_counts = np.unique(vl_sg_tok, return_counts=True)
#count_sort_ind = np.argsort(-vl_pl_tok_counts)
#vl_pl_tok_unq[count_sort_ind]
#vl_pl_tok_counts[count_sort_ind]

#%%
WORD_EMB = V[word_tok_unq]
SG_EMB = V[vl_sg_tok]
PL_EMB = V[vl_pl_tok]

#%%


def get_logs(hidden_state):
    word_log = WORD_EMB @ hidden_state
    sg_log = SG_EMB @ hidden_state
    pl_log = PL_EMB @ hidden_state
    return word_log, sg_log, pl_log

def get_probs(hidden_state):
    word_log, sg_log, pl_log = get_logs(hidden_state)
    all_logits = np.hstack([word_log, sg_log, pl_log])
    all_probs = softmax(all_logits)
    word_probs = all_probs[:word_log.shape[0]]
    sg_end = word_log.shape[0] + sg_log.shape[0]
    sg_probs = all_probs[word_log.shape[0]:sg_end]
    pl_probs = all_probs[sg_end:]
    lemma_probs = sg_probs + pl_probs
    full_probs = np.hstack([word_probs,lemma_probs])
    return full_probs, sg_probs, pl_probs

def normalize_pairs(sg, pl):
    base_pair = np.vstack((sg, pl)).T
    base_pair_Z = np.sum(base_pair,axis=1)
    base_pair_probs = np.vstack([
        np.divide(base_pair[:,0], base_pair_Z),
        np.divide(base_pair[:,1], base_pair_Z)]
    )
    return base_pair_probs


#%% Loading Run
RUN_OUTPUT = "../out/run_output/bert-base-uncased/20230216/run_model_bert-base-uncased_k_1_n_20000_0_3.pkl"

with open(RUN_OUTPUT, 'rb') as f:      
    run = pickle.load(f)

P = run["diag_rlace"]["P_acc"]


idx = np.arange(0, h.shape[0])
np.random.shuffle(idx)
ind = idx[:200]
fth_kls = []
er_kls = []
for i in ind:
    base = h[i]
    base_probs, base_sg, base_pl = get_probs(base)
    er_probs, er_sg, er_pl = get_probs(P @ base)

    fth_kl = np.sum(kl_div(base_probs, er_probs))
    fth_kls.append(fth_kl)

    base_pair_probs = normalize_pairs(base_sg, base_pl)
    er_pair_probs = normalize_pairs(er_sg, er_pl)
    obs_er_kls = []
    for base_pair, er_pair in zip(base_pair_probs, er_pair_probs):
        obs_er_kls.append(np.sum(kl_div(base_pair, er_pair)))
    er_kls.append(np.median(obs_er_kls))


#%%
scipy.stats.describe(fth_kls)

#%%
scipy.stats.describe(er_kls)

#%%