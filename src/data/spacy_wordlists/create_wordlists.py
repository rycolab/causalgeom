#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

from tqdm import tqdm
import pickle
import spacy
from datasets import load_dataset

sys.path.append('../../')

from paths import HF_CACHE, OUT, DATASETS

#%% UD FR DATA FOR REFERENCE
DATA_FILE = os.path.join(DATASETS, "preprocessed/ud_fr_gsd/train.pkl")
with open(DATA_FILE, 'rb') as f:      
    sent_data = pickle.load(f)

#%%
language = "fr"
OUT_DIR = os.path.join(OUT, f"wordlist")
OUT_FILE = os.path.join(OUT_DIR, f"{language}_new.pkl")

with open(OUT_FILE, 'rb') as f:
    token_data = pickle.load(f)

# %%
def find_substr(fullstr, substr):
    substr_ind = fullstr.find(substr)
    if substr_ind > -1:
        substr_ind = substr_ind + len(substr)
        return fullstr[substr_ind:substr_ind+4]
    else:
        return ""

def get_gender_number(morphodict):
    gennumstr = max(morphodict, key=morphodict.get)
    gender = find_substr(gennumstr, "Gender=")
    number = find_substr(gennumstr, "Number=")
    return gennumstr, gender, number

def get_lemma(lemmadict):
    lowerlemmadict = {}
    for k, v in lemmadict.items():
        lowerv = lemmadict.get(k.lower(), 0)
        lowerlemmadict[k] = lowerv + v
    maxkey = max(lowerlemmadict, key=lowerlemmadict.get)
    return maxkey, lowerlemmadict[maxkey]

ADJ_dict = {}
adj_list = []
for k, v in token_data.items():
    for kp, vp in v["token_tag"].items():
        if kp.strip()=="ADJ" and (vp / v["count"]) > .5 and v["count"] > 500:
            gennumstr, gender, number = get_gender_number(v["token_morph"])
            lemma, lemmacount = get_lemma(v["token_lemma"])
            res = dict(
                adj=k,
                lemma=lemma,
                count=vp, 
                gender=gender, 
                number=number, 
                gender_ratio=v["token_morph"][gennumstr] / vp,
                lemma_ratio=lemmacount/vp
            )
            ADJ_dict[k] = res
            adj_list.append(res)
        else:
            continue

#%%
gender_vals = set()
number_vals = set()
for k, v in ADJ_dict.items():
    gender_vals.update([v["gender"]])
    number_vals.update([v["number"]])

#%%
import pandas as pd
df = pd.DataFrame(adj_list)

mascpl = df[(df["gender"] == "Masc") & (df["number"] == "Plur")]
mascsg = df[(df["gender"] == "Masc") & (df["number"] == "Sing")]
masc = df[(df["gender"] == "Masc") & (df["number"] == "")]
fempl = df[(df["gender"] == "Fem|") & (df["number"] == "Plur")]
femsg = df[(df["gender"] == "Fem|") & (df["number"] == "Sing")]
fem = df[(df["gender"] == "Fem|") & (df["number"] == "")]
pl = df[(df["gender"] == "") & (df["number"] == "Plur")]
sg = df[(df["gender"] == "") & (df["number"] == "Sing")]

#%%
vc = mascsg["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
mascsg_dedup = mascsg[~mascsg["lemma"].isin(dup_lemmas)]

vc = femsg["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
femsg_dedup = femsg[~femsg["lemma"].isin(dup_lemmas)]

sg_mascfem = pd.merge(
    left=mascsg_dedup,
    right=femsg_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

#%%
vc = mascpl["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
mascpl_dedup = mascpl[~mascpl["lemma"].isin(dup_lemmas)]

vc = fempl["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
fempl_dedup = fempl[~fempl["lemma"].isin(dup_lemmas)]

pl_mascfem = pd.merge(
    left=mascpl_dedup,
    right=fempl_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

#%%
vc = masc["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
masc_dedup = masc[~masc["lemma"].isin(dup_lemmas)]

vc = fem["lemma"].value_counts()
dup_lemmas = vc[vc>1].index.to_numpy()
fem_dedup = fem[~fem["lemma"].isin(dup_lemmas)]

na_mascfem = pd.merge(
    left=masc_dedup,
    right=fem_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)
#%%
# TODO: 
# - merged sg, pl masc and fem lists
# - clean up said lists -- tgt list done
# - take all other tokens and filter out the ones that are in the tgt list
# done :)
