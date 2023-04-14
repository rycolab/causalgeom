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

from paths import OUT, UNIMORPH_ENG, DATASETS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% PARAMETERS -- note that there is no difference for this portion between different models (same dataset)
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "linzen"
DATASET = os.path.join(DATASETS, f"processed/{DATASET_NAME}/{DATASET_NAME}_{MODEL_NAME}_verbs.pkl")

# MANUAL_VERBLIST is a cleaned up version of the full verb list
# with manual drops.
MANUAL_VERBLIST_PATH = os.path.join(DATASETS, f"processed/{DATASET_NAME}/word_lists/linzen_verb_list_drop.csv")
OUTFILE = os.path.join(DATASETS, f"processed/{DATASET_NAME}/word_lists/verb_list_final.pkl")
SG_PL_OUTFILE = os.path.join(DATASETS, f"processed/{DATASET_NAME}/word_lists/sg_pl_prob.pkl")

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
manual_verblist = pd.read_csv(MANUAL_VERBLIST_PATH, index_col=0)
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

#%% RELATIVE COUNTS PAIRWISE
vpc_pos = verbpairs.groupby(["sverb", "pverb", "verb_pos"])["count"].sum().reset_index()
vpc_pos_tab = vpc_pos.pivot(index=["sverb", "pverb"], columns="verb_pos", values="count").reset_index()
vpc_pos_tab.fillna(value=0, inplace=True)
vpc_pos_tab["total"] = vpc_pos_tab["VBZ"] + vpc_pos_tab["VBP"]
vpc_pos_tab["p_sg"] = vpc_pos_tab["VBZ"] / vpc_pos_tab["total"]
vpc_pos_tab["p_pl"] = vpc_pos_tab["VBP"] / vpc_pos_tab["total"]

vpc_pos_tab.drop(["VBP", "VBZ", "total"], axis=1, inplace=True)

#%%
verblist_final_p = pd.merge(
    left=verblist_final,
    right=vpc_pos_tab,
    on=["sverb", "pverb"],
    how="inner"
)
assert verblist_final.shape[0] == verblist_final_p.shape[0]

#%%
with open(OUTFILE, 'wb') as f:
    pickle.dump(verblist_final_p, f, protocol=pickle.HIGHEST_PROTOCOL)

#%% RELATIVE COUNTS OVERALL
sg_pl = verbpairs["verb_pos"].value_counts()
sg_pl_prob = sg_pl / sg_pl.sum()

with open(SG_PL_OUTFILE, 'wb') as f:
    pickle.dump(sg_pl_prob, f, protocol=pickle.HIGHEST_PROTOCOL)

