#TODO:
# - once verb data is re-output, update this to get relative counts
# - debug the removal of the tokens

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


sys.path.append('..')
#sys.path.append('./src/')

from paths import OUT, UNIMORPH_ENG, HF_CACHE

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% PARAMETERS -- note that there is no difference for this portion between different models (same dataset)
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "linzen"
DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_verbs.pkl"


#%%#################
# Unimorph         #
####################
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

#%% FACT - FOIL PAIRS UNIMOPRH
#vc = df[((df["sg"] == 1) | (df["pl"] == 1))]["lemma"].value_counts()
#fact_foil_pairs = vc[vc == 2].index
sg_verbs = df[df["sg"]==1][["lemma", "word", "morph"]]
pl_verbs = df[df["pl"]==1][["lemma", "word", "morph"]]
all_verbs = pd.merge(
    sg_verbs,
    pl_verbs,
    on="lemma",
    how="outer",
    suffixes=("_sg", "_pl")
)
all_verbs.dropna(inplace=True)

#%%###########
# DATASET    #
##############
with open(DATASET, 'rb') as f:      
    verbpairs = pd.DataFrame(pickle.load(f), columns = ["sverb", "sverb_tok", "pverb", "pverb_tok", "verb_pos"])

#verbpairs["sverb_tok_str"] = verbpairs["sverb_tok"].astype(str)
#verbpairs["pverb_tok_str"] = verbpairs["pverb_tok"].astype(str)
#verbpairs["pverb_tok"] = verbpairs["pverb_tok"].apply(lambda x: str(x))

#verbpairs.drop_duplicates(
#    subset=["sverb", "sverb_tok_str", "pverb", "pverb_tok_str"], 
#    inplace=True
#)
verbpairs["count"] = 1
vpc = verbpairs.groupby(["sverb", "pverb"])["count"].sum().reset_index()

#TODO: make the relative counts
vpc_pos = verbpairs.groupby(["sverb", "pverb", "verb_pos"])["count"].sum().reset_index()

# dropping specific wrong obs that lead to dups
#vpc["sverb"].value_counts()
#vpc["pverb"].value_counts()

vpc.drop(vpc[vpc["sverb"] == vpc["pverb"]].index, inplace=True)

#%%
verbs_merged = pd.merge(
    left=all_verbs,
    right=vpc,
    left_on=["word_sg", "word_pl"],
    right_on=["sverb", "pverb"],
    how="right"
)
#%%
verbs_merged_check = pd.merge(
    left=all_verbs[["word_sg", "word_pl"]],
    right=verbs_merged,
    left_on=["word_pl", "word_sg"],
    right_on=["sverb", "pverb"],
    how="right",
    suffixes=("_flipcheck", "")
)
verbs_merged_check.drop(
    verbs_merged_check[verbs_merged_check["word_sg_flipcheck"].notnull()].index,
    inplace=True
)

#%%
verbs_merged_check["uni"] = verbs_merged_check["word_sg"].notnull()*1
verblist = verbs_merged_check[["sverb", "pverb", "count", "uni"]]
#verblist.to_csv("../../out/linzen_verb_list.csv")

#%%
manual_verblist = pd.read_csv("../../datasets/processed/linzen_verb_list/linzen_verb_list_drop.csv", index_col=0)
manual_verblist["drop"].fillna(value=0, inplace=True)

verblist_drop = pd.merge(
    left=verblist,
    right=manual_verblist[["sverb", "pverb", "drop"]],
    on=["sverb", "pverb"],
    how="inner"
)

assert verblist.shape[0] == verblist_drop.shape[0]

#%% cleaning up token lists
#def str_to_list(tok_str):
#    strlist = tok_str[1:-1].split(",")
#    return [int(x) for x in strlist]

#verblist_drop["sverb_tok"] = verblist_drop["sverb_tok_str"].apply(str_to_list)
#verblist_drop["pverb_tok"] = verblist_drop["pverb_tok_str"].apply(str_to_list)

#%% final dataset
verblist_final = verblist_drop.drop(
    verblist_drop[verblist_drop["drop"]==1].index)[
        ["sverb", "pverb"]]

OUTFILE = "../../datasets/processed/linzen_verb_list/linzen_verb_list_final.pkl"
with open(OUTFILE, 'wb') as file:
    pickle.dump(verblist_final, file, protocol=pickle.HIGHEST_PROTOCOL)

# %%
