#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from transformers import GPT2TokenizerFast, GPT2LMHeadModel

#import torch
#from torch.utils.data import DataLoader, Dataset
#from abc import ABC

sys.path.append('..')
#sys.path.append('./src/')

from paths import OUT, UNIMORPH_ENG, HF_CACHE

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

# TODO:
# - check that all fact/foil pairs in the data appear in this selection
#    w 2 words per lemma
# - tokenize all of the word and keep only the first token
# - is/are are not in unimoprh

#%% PARAMETERS
MODEL_NAME = "gpt2"
DATASET_NAME = "linzen"
DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_verbs.pkl"

#%%#################
# Unimorph         #
####################

#%%
def create_bin(vals, morph):
    return all(i in morph for i in vals)

#%% LOAD UNIMORPH
data = []
with open(UNIMORPH_ENG) as f:
    tsv_file = csv.reader(f, delimiter="\t")
    for line in tsv_file:
        morph = line[2]
        morph_list = morph.split(";")
        data.append(dict(
            lemma = line[0],
            word = line[1],
            morph = morph,
            #sg = create_bin(["V", "PRS", "3", "SG"], morph),
            #pl = create_bin(["V", "NFIN", "IMP+SBJV"], morph)
            sg = ["V", "PRS", "3", "SG"] == morph_list,
            pl = ["V", "NFIN", "IMP+SBJV"] == morph_list
        ))

# %%
df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
del data

#%% TOKENIZE UNIMORPH
TOKENIZER = GPT2TokenizerFast.from_pretrained(
    MODEL_NAME, model_max_length=512
)

def tokenize_word(word):
    return TOKENIZER(word)["input_ids"]

df["input_ids_word"] = df["word"].apply(tokenize_word)
df["first_id_word"] = df["input_ids_word"].apply(lambda x: int(x[0]))

#%% FACT - FOIL PAIRS UNIMOPRH
#vc = df[((df["sg"] == 1) | (df["pl"] == 1))]["lemma"].value_counts()
#fact_foil_pairs = vc[vc == 2].index
sg_verbs = df[df["sg"]==1][["lemma", "word", "morph", "input_ids_word", "first_id_word"]]
pl_verbs = df[df["pl"]==1][["lemma", "word", "morph", "input_ids_word", "first_id_word"]]
all_verbs = pd.merge(
    sg_verbs,
    pl_verbs,
    on="lemma",
    how="outer",
    suffixes=("_sg", "_pl")
)


#%%###########
# DATASET    #
##############
with open(DATASET, 'rb') as f:      
    verbpairs = pd.DataFrame(pickle.load(f), columns = ["verb_tok", "iverb_tok"])

verbpairs["verb_1"] = verbpairs["verb_tok"].apply(lambda x: len(x))
verbpairs["iverb_1"] = verbpairs["iverb_tok"].apply(lambda x: len(x))
verbpairs.drop(verbpairs[(verbpairs["verb_1"] != 1) | (verbpairs["iverb_1"] != 1)].index, inplace=True)
verbpairs["verb"] = verbpairs["verb_tok"].apply(lambda x: x[0])
verbpairs["iverb"] = verbpairs["iverb_tok"].apply(lambda x: x[0])

cleanpairs = verbpairs[["verb", "iverb"]].drop_duplicates()

#%%

unique_list = df["first_id_word"].unique()
#%%
MODEL = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME, 
    cache_dir=HF_CACHE
)

V = MODEL.lm_head.weight.detach().numpy()

#%%
h = np.random.uniform(size=(768,))
from scipy.special import softmax

probs = softmax(V @ h)

#%%

