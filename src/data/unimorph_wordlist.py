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

from transformers import GPT2TokenizerFast, BertTokenizerFast

sys.path.append('..')
#sys.path.append('./src/')

from paths import OUT, UNIMORPH_ENG, HF_CACHE

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%% ARGS
VERBLIST_PATH = "../../datasets/processed/linzen_word_lists/linzen_verb_list_final.pkl"

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
#if MODEL_NAME == "gpt2":
#    TOKENIZER = GPT2TokenizerFast.from_pretrained(
#        MODEL_NAME, model_max_length=512
#    )
#elif MODEL_NAME == "bert-base-uncased":
#    TOKENIZER = BertTokenizerFast.from_pretrained(
#        MODEL_NAME, model_max_length=512
#    )
#else:
#    logging.warn("NO TOKENIZER")

#def tokenize_word(word, add_space=False):
#    if add_space:
#        return TOKENIZER(" "+word)["input_ids"]
#    else:
#        return TOKENIZER(word)["input_ids"]

#df["input_ids_word"] = df["word"].apply(lambda x: tokenize_word(x)[1:-1])
#df["input_ids_word_spc"] = df["word"].apply(lambda x: tokenize_word(x, add_space=True)[1:-1])
#df["ntokens"] = df["input_ids_word"].apply(lambda x: len(x))
#df["ntokens_spc"] = df["input_ids_word_spc"].apply(lambda x: len(x))
#df["first_id_word"] = df["input_ids_word"].apply(lambda x: int(x[0]))

#%% 
with open(VERBLIST_PATH, 'rb') as f:      
    verblist = pickle.load(f)

verblist_all = pd.concat([verblist["sverb"], verblist["pverb"]], axis=0)
verblist_all.name = "verb"
verblist_all = verblist_all.reset_index(drop=False)
verblist_all["verb_flag"] = 1
verblist_all = verblist_all[["verb", "verb_flag"]]

#%% WORDLIST
dfv = pd.merge(
    left=df,
    right=verblist_all,
    left_on="word",
    right_on="verb",
    how="left"
)

wordlist = dfv[#(dfv["ntokens"] == 1) & 
                (dfv["verb"].isnull()) & 
                (dfv["sg"]!=True) & 
                (dfv["pl"]!=True)]

wordlist = wordlist[["word"]]

#wordlist["input_id"] = wordlist["input_ids_word"].apply(lambda x: x[0])
#wordlist["input_id_spc"] = wordlist["input_ids_word_spc"].apply(lambda x: x[0])
#wordlist[["word", "input_id"]].to_csv("../../datasets/processed/wordlists/bert-base-uncased_wordlist.csv")
wordlist.to_csv("../../datasets/processed/linzen_word_lists/linzen_wordlist.csv")

"""
#%%

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

#%% OTHER UNIMORPH FILES
uni_dir = os.path.dirname(UNIMORPH_ENG)
uni_seg = os.path.join(
    uni_dir, "eng.segmentations"
)
uni_args = os.path.join(
    uni_dir, "eng.args"
)
uni_derivs = os.path.join(
    uni_dir, "eng.derivations.tsv"
)

#%%
data = []
with open(uni_seg) as f:
    tsv_file = csv.reader(f, delimiter="\t")
    for line in tsv_file:
        assert len(line) == 4, f"Line: {line}"
        if "|" in line[2]:
            morph_list = line[2].split("|")
            morph_main = morph_list[0]
            morph_tags = morph_list[1].split(";")
        else:
            morph_main = None
            morph_tags = morph_list[1].split(";")
        data.append(dict(
            lemma = line[0],
            word = line[1],
            morph_main = morph_main,
            morph_tags = morph_tags,
            #sg = create_bin(["V", "PRS", "3", "SG"], morph),
            #pl = create_bin(["V", "NFIN", "IMP+SBJV"], morph)
            sg = ["V", "PRS", "3", "SG"] == morph_tags,
            pl = ["V", "NFIN", "IMP+SBJV"] == morph_tags
        ))

#%%
data = []
with open(uni_args) as f:
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
        
args_df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
del data

#%%
data = []
with open(uni_derivs) as f:
    tsv_file = csv.reader(f, delimiter="\t")
    for line in tsv_file:
        morph = line[2]
        morph_list = morph.split(";")
        data.append(dict(
            word = line[0],
            deriv = line[1],
            morph = morph,
            suffix = line[3],
            #sg = create_bin(["V", "PRS", "3", "SG"], morph),
            #pl = create_bin(["V", "NFIN", "IMP+SBJV"], morph)
            sg = ["V", "PRS", "3", "SG"] == morph_list,
            pl = ["V", "NFIN", "IMP+SBJV"] == morph_list
        ))
        
deriv_df = pd.DataFrame(data)
deriv_df.drop_duplicates(inplace=True)
del data
"""