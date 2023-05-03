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
OUT_FILE = os.path.join(OUT_DIR, f"{language}_new_0.pkl")

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

def format_gender(gender):
    if gender == "Masc":
        return "m"
    elif gender == "Fem|":
        return "f"
    else:
        return ""

def format_number(number):
    if number == "Sing":
        return "sg"
    elif number == "Plur":
        return "pl"
    else:
        return ""

def get_gender_number(morphodict):
    gennumstr = max(morphodict, key=morphodict.get)
    gender = format_gender(find_substr(gennumstr, "Gender="))
    number = format_number(find_substr(gennumstr, "Number="))
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
other_list = []
for k, v in token_data.items():
    for kp, vp in v["token_tag"].items():
        if kp.strip()=="ADJ" and (vp / v["count"]) > .5 and v["count"] > 50:
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
    other_list.append(
        dict(
            word=k,
            count=v["count"]
        )
    )

gender_vals = set()
number_vals = set()
for k, v in ADJ_dict.items():
    gender_vals.update([v["gender"]])
    number_vals.update([v["number"]])

#%%
import pandas as pd
df = pd.DataFrame(adj_list)

mascpl = df[(df["gender"] == "m") & (df["number"] == "pl")]
mascsg = df[(df["gender"] == "m") & (df["number"] == "sg")]
masc = df[(df["gender"] == "m") & (df["number"] == "")]
fempl = df[(df["gender"] == "f") & (df["number"] == "pl")]
femsg = df[(df["gender"] == "f") & (df["number"] == "sg")]
fem = df[(df["gender"] == "f") & (df["number"] == "")]
pl = df[(df["gender"] == "") & (df["number"] == "pl")]
sg = df[(df["gender"] == "") & (df["number"] == "sg")]


def dedup_lemma(df):
    vc = df["lemma"].value_counts()
    dup_lemmas = vc[vc>1].index.to_numpy()
    df_dedup = df[~df["lemma"].isin(dup_lemmas)]
    df_dup = df[df["lemma"].isin(dup_lemmas)]
    return df_dedup, df_dup

mascsg_dedup, mascsg_dup = dedup_lemma(mascsg)
femsg_dedup, femsg_dup = dedup_lemma(femsg)
mascpl_dedup, mascpl_dup = dedup_lemma(mascpl)
fempl_dedup, fempl_dup = dedup_lemma(fempl)
masc_dedup, masc_dup = dedup_lemma(masc)
fem_dedup, fem_dup = dedup_lemma(fem)

sg_mascfem = pd.merge(
    left=mascsg_dedup,
    right=femsg_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

pl_mascfem = pd.merge(
    left=mascpl_dedup,
    right=fempl_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

na_mascfem = pd.merge(
    left=masc_dedup,
    right=fem_dedup,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

#%%
# TODO: 
# - once enough data gathered, clean up said lists
# - handle dup lists manually
# - take all other tokens and filter out the ones that are in the tgt list
# done :)

#%%
def filter_adjs(df, drop_threshold=5000):
    filtered_df = df.dropna(subset=["adj_masc", "adj_fem"], inplace=False)
    drop_df = filtered_df.drop(
        filtered_df[(filtered_df["count_masc"] < drop_threshold) & 
                (filtered_df["count_fem"] < drop_threshold)].index, 
        axis=0
    )
    return drop_df

#%%
drop_threshold = 5000
sglist = filter_adjs(sg_mascfem, drop_threshold)
pllist = filter_adjs(pl_mascfem, drop_threshold)
nalist = filter_adjs(na_mascfem, drop_threshold)

lemma_list = pd.concat([sglist, pllist, nalist], axis=0)
lemma_list["total_count"] = lemma_list["count_masc"] + lemma_list["count_fem"]
lemma_list["p_0"] = lemma_list["count_masc"] / lemma_list["total_count"]
lemma_list["p_1"] = lemma_list["count_fem"] / lemma_list["total_count"]
lemma_final = lemma_list[["adj_masc", "adj_fem", "p_0", "p_1"]] # "count_masc", "count_fem", "number_masc"]]
lemma_final.rename(
    {"adj_masc":"lemma_0",
     "adj_fem": "lemma_1"}, 
    axis=1,
    inplace=True
)

# %%
#init_list_outpath = os.path.join(DATASETS, "processed/ud_fr_gsd/word_lists/init_adjlist.tsv")
#final_list.to_csv(init_list_outpath, sep="\t")
adj_list_outpath = os.path.join(DATASETS, "processed/fr/word_lists/adj_list.tsv")
lemma_final.to_csv(adj_list_outpath, sep="\t")


# %%
other_df = pd.DataFrame(other_list)

other_sub = other_df[other_df["count"]>10000]
other_sub.sort_values(by="count", inplace=True, ascending=False)

#%%
masc_adj_list = lemma_final["lemma_0"].unique()
fem_adj_list = lemma_final["lemma_1"].unique()

other_sub["adj_flag"] = other_sub["word"].apply(lambda x: 1 if ((x in masc_adj_list) or (x in fem_adj_list)) else 0)
other_sub.drop(other_sub[other_sub["adj_flag"]==1].index, axis=0, inplace=True)
#%%
other_list_outpath = os.path.join(DATASETS, "processed/fr/word_lists/other_list.tsv")
other_sub[["word"]].to_csv(other_list_outpath, sep="\t")

# %%
