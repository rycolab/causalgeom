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

sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Arguments        #
####################
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "linzen"
WORDLIST_PATH = os.path.join(DATASETS, "processed/linzen_word_lists/linzen_wordlist.csv")
VERBLIST_PATH = os.path.join(DATASETS, "processed/linzen_word_lists/linzen_verb_list_final.pkl")

if MODEL_NAME == "gpt2":
    DATASET = os.path.join(DATASETS, f"processed/{DATASET_NAME}_{MODEL_NAME}_ar.pkl")
elif MODEL_NAME == "bert-base-uncased":
    DATASET = os.path.join(DATASETS, f"processed/{DATASET_NAME}_{MODEL_NAME}_masked.pkl")
else:
    DATASET = None

with open(DATASET, 'rb') as f:      
    data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])
    h = data["h"]
    del data

#%%
TOKENIZER = get_tokenizer(MODEL_NAME)
V = get_V(MODEL_NAME)

#%%#################
# Wordlist         #
####################
wl = pd.read_csv(WORDLIST_PATH, index_col=0)
wl.drop_duplicates(inplace=True)

def tokenize_word(word, add_space=False):
    if add_space and type(word) == str:
        return TOKENIZER(" "+word)["input_ids"]
    elif type(word) == str:
        return TOKENIZER(word)["input_ids"]
    else:
        return []

wl["input_ids_word"] = wl["word"].apply(lambda x: tokenize_word(x)[1:-1])
#df["input_ids_word_spc"] = df["word"].apply(lambda x: tokenize_word(x, add_space=True)[1:-1])
wl["ntokens"] = wl["input_ids_word"].apply(lambda x: len(x))
#df["ntokens_spc"] = df["input_ids_word_spc"].apply(lambda x: len(x))
wl_1tok = wl[wl["ntokens"]==1]
wl_1tok["first_id_word"] = wl_1tok["input_ids_word"].apply(lambda x: int(x[0]))

word_tok = wl_1tok["first_id_word"].to_numpy()
word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
count_sort_ind = np.argsort(-word_tok_counts)
word_tok_unq[count_sort_ind]
word_tok_counts[count_sort_ind]

#word_tok_unq 

#%%#################
# Verblist         #
####################
with open(VERBLIST_PATH, 'rb') as f:      
    vl = pickle.load(f)

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
    return all_probs, word_probs, sg_probs, pl_probs

def get_merged_probs(word_probs, sg_probs, pl_probs):
    lemma_merged = sg_probs + pl_probs
    all_merged = np.hstack([word_probs,lemma_merged])
    return lemma_merged, all_merged

def normalize_pairs(sg, pl):
    base_pair = np.vstack((sg, pl)).T
    base_pair_Z = np.sum(base_pair,axis=1)
    base_pair_probs = np.vstack([
        np.divide(base_pair[:,0], base_pair_Z),
        np.divide(base_pair[:,1], base_pair_Z)]
    ).T
    return base_pair_probs

#%% Loading Run
RUN_OUTPUT = os.path.join(OUT, "run_output/bert-base-uncased/20230216/run_model_bert-base-uncased_k_1_n_20000_0_3.pkl")

with open(RUN_OUTPUT, 'rb') as f:      
    run = pickle.load(f)

P = run["diag_rlace"]["P_acc"]

idx = np.arange(0, h.shape[0])
np.random.shuffle(idx)
ind = idx[:200]
kls_all_split = []
kls_all_merged = []
kls_words = []
kls_tgt_split = []
kls_tgt_merged = []
er_kls = []

pbar = tqdm(ind)
pbar.set_description("Testing on hidden states")
for i in pbar:
    base = h[i]
    base_all, base_words, base_sg, base_pl = get_probs(base)
    er_all, er_words, er_sg, er_pl = get_probs(P @ base)

    base_lemma_split = np.hstack([base_sg, base_pl])
    er_lemma_split = np.hstack([er_sg, er_pl])

    base_lemma_merged, base_all_merged = get_merged_probs(
        base_words, base_sg, base_pl
    )
    er_lemma_merged, er_all_merged = get_merged_probs(
        er_words, er_sg, er_pl
    )
    
    kls_all_split.append(np.mean(kl_div(base_all, er_all)))
    kls_all_merged.append(np.mean(kl_div(base_all_merged, er_all_merged)))
    kls_words.append(np.mean(kl_div(base_words, er_words)))
    kls_tgt_split.append(np.mean(kl_div(base_lemma_split, er_lemma_split)))
    kls_tgt_merged.append(np.mean(kl_div(base_lemma_merged, er_lemma_merged)))
    
    base_pair_probs = normalize_pairs(base_sg, base_pl)
    er_pair_probs = normalize_pairs(er_sg, er_pl)
    obs_er_kls = []
    for base_pair, er_pair in zip(base_pair_probs, er_pair_probs):
        obs_er_kls.append(np.mean(kl_div(base_pair, er_pair)))
    er_kls.append(np.mean(obs_er_kls))


#%%
kls_all_split_stats = pd.DataFrame(kls_all_split).describe()
kls_all_merged_stats = pd.DataFrame(kls_all_merged).describe()
kls_words_stats = pd.DataFrame(kls_words).describe()
kls_tgt_split_stats = pd.DataFrame(kls_tgt_split).describe()
kls_tgt_merged_stats = pd.DataFrame(kls_tgt_merged).describe()
er_kls_stats = pd.DataFrame(er_kls).describe()

stats_list = [kls_all_split_stats, kls_all_merged_stats, kls_words_stats, kls_tgt_split_stats, kls_tgt_merged_stats, er_kls_stats]
#%%
stats = pd.concat(stats_list, axis=1)
stats.columns = [
    "all_split", "all_merged", "words", "tgt_split", "tgt_merged", "er_kl"]

stats.to_csv(os.path.join(OUT, "stats.csv"))