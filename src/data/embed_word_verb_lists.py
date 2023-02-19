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
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Arguments        #
####################
MODEL_NAME = "gpt2"
DATASET_NAME = "linzen"
WORDLIST_PATH = os.path.join(DATASETS, "processed/linzen_word_lists/linzen_wordlist.csv")
VERBLIST_PATH = os.path.join(DATASETS, "processed/linzen_word_lists/linzen_verb_list_final.pkl")

WORD_EMB_OUTFILE = os.path.join(DATASETS, f"processed/linzen_word_lists/{MODEL_NAME}_word_embeds.npy")
VERB_P_OUTFILE = os.path.join(DATASETS, f"processed/linzen_word_lists/{MODEL_NAME}_verb_p.npy")
SG_EMB_OUTFILE = os.path.join(DATASETS, f"processed/linzen_word_lists/{MODEL_NAME}_sg_embeds.npy")
PL_EMB_OUTFILE = os.path.join(DATASETS, f"processed/linzen_word_lists/{MODEL_NAME}_pl_embeds.npy")

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {MODEL_NAME}")

#%%#################
# Loading        #
####################
if MODEL_NAME == "gpt2":
    MASKED=False
elif MODEL_NAME == "bert-base-uncased":
    MASKED=True
else:
    raise ValueError("Incorrect model name")
    MASKED = None

TOKENIZER = get_tokenizer(MODEL_NAME)
V = get_V(MODEL_NAME)

#%%#################
# Helpers         #
####################
def tokenize_word(tokenizer, word, masked):
    if type(word) == str and masked:
        return tokenizer(" "+word)["input_ids"][1:-1]
    elif type(word) == str:
        return tokenizer(word)["input_ids"]
    else:
        return []

#%%#################
# Wordlist         #
####################

"""
def tokenize_words(tokenizer, words, masked):
    word_df = pd.DataFrame(words)
    
    word_df["input_ids"] = word_df["word"].apply(
        lambda x: tokenize_word(tokenizer, x, masked=masked))
    word_df["ntokens"] = word_df["input_ids"].apply(lambda x: len(x))

    word_df.drop(word_df[word_df["ntokens"]!=1].index, axis=0, inplace=True)
    word_df["first_id_word"] = word_df["input_ids"].apply(lambda x: int(x[0]))

    word_tok = word_df["first_id_word"].to_numpy()
    word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
    return word_tok_unq

def get_unique_word_tokens(tokenizer, masked):
    wl = pd.read_csv(WORDLIST_PATH, index_col=0)
    wl.drop_duplicates(inplace=True)

    single_tok_words = tokenize_words(
        tokenizer, wl["word"], masked=masked)
    return single_tok_words

word_toks = get_unique_word_tokens(TOKENIZER, MASKED)
"""
wl = pd.read_csv(WORDLIST_PATH, index_col=0)
wl.drop_duplicates(inplace=True)

wl["input_ids_word"] = wl["word"].apply(lambda x: tokenize_word(TOKENIZER, x, MASKED))
#df["input_ids_word_spc"] = df["word"].apply(lambda x: tokenize_word(x, add_space=True)[1:-1])
wl["ntokens"] = wl["input_ids_word"].apply(lambda x: len(x))
#df["ntokens_spc"] = df["input_ids_word_spc"].apply(lambda x: len(x))
wl_1tok = wl[wl["ntokens"]==1]
wl_1tok["first_id_word"] = wl_1tok["input_ids_word"].apply(lambda x: int(x[0]))

word_tok = wl_1tok["first_id_word"].to_numpy()
word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
#count_sort_ind = np.argsort(-word_tok_counts)
#word_tok_unq[count_sort_ind]
#word_tok_counts[count_sort_ind]

logging.info(f"Single token word list of length: {len(word_tok_unq)}")

word_emb = V[word_tok_unq]
np.save(WORD_EMB_OUTFILE, word_emb)

logging.info(f"Tokenized and exported word embeds to: {WORD_EMB_OUTFILE}")
"""

(word_tok_unq == res).all()
"""

#%%#################
# Verblist         #
####################
with open(VERBLIST_PATH, 'rb') as f:      
    vl = pickle.load(f)

for col in ["sverb", "pverb"]:
    vl[f"{col}_input_ids"] = vl[col].apply(
        lambda x: tokenize_word(TOKENIZER, x, masked=MASKED))
    vl[f"{col}_ntokens"] = vl[f"{col}_input_ids"].apply(
        lambda x: len(x))

vl["1tok"] = (vl["sverb_ntokens"] == 1) & (vl["pverb_ntokens"]==1)

vl.drop(vl[vl["1tok"] != True].index, inplace=True)

vl["sverb_tok"] = vl["sverb_input_ids"].apply(lambda x: x[0])
vl["pverb_tok"] = vl["pverb_input_ids"].apply(lambda x: x[0])

# Exporting P(sg), P(pl)
verb_p = vl[["p_sg", "p_pl"]].to_numpy()
np.save(VERB_P_OUTFILE, verb_p)
logging.info(f"Exported single token verb probs to: {VERB_P_OUTFILE}")

vl_sg_tok = vl["pverb_tok"].to_numpy()
logging.info(f"Single token sg verb list of length: {len(vl_sg_tok)}")

sg_emb = V[vl_sg_tok]
np.save(SG_EMB_OUTFILE, sg_emb)
logging.info(f"Tokenized and exported sg verb embeds to: {SG_EMB_OUTFILE}")
#vl_sg_tok_unq, vl_sg_tok_counts = np.unique(vl_sg_tok, return_counts=True)
#count_sort_ind = np.argsort(-vl_sg_tok_counts)
#vl_sg_tok_unq[count_sort_ind]
#vl_sg_tok_counts[count_sort_ind]

vl_pl_tok = vl["pverb_tok"].to_numpy()
logging.info(f"Single token pl verb list of length: {len(vl_pl_tok)}")

pl_emb = V[vl_pl_tok]
np.save(PL_EMB_OUTFILE, pl_emb)
logging.info(f"Tokenized and exported pl verb embeds to: {PL_EMB_OUTFILE}")

#vl_pl_tok_unq, vl_pl_tok_counts = np.unique(vl_sg_tok, return_counts=True)
#count_sort_ind = np.argsort(-vl_pl_tok_counts)
#vl_pl_tok_unq[count_sort_ind]
#vl_pl_tok_counts[count_sort_ind]
