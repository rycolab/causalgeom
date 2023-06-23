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
import pandas as pd

#sys.path.append('../../')
sys.path.append('./src/')

from paths import HF_CACHE, OUT, DATASETS

#%% UD FR DATA FOR REFERENCE
#DATA_FILE = os.path.join(DATASETS, "preprocessed/ud_fr_gsd/train.pkl")
#with open(DATA_FILE, 'rb') as f:      
#    sent_data = pickle.load(f)

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
total_count = 0
for k, v in token_data.items():
    token_tags = dict(sorted(v["token_tag"].items(), reverse=True))
    token_count = v["count"]
    for kp, vp in token_tags.items():
        if kp.strip()=="ADJ" and (vp / token_count) > .5 and token_count > 50:
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
            count=token_count
        )
    )
    total_count+=token_count

gender_vals = set()
number_vals = set()
for k, v in ADJ_dict.items():
    gender_vals.update([v["gender"]])
    number_vals.update([v["number"]])

#%%

df = pd.DataFrame(adj_list)
df.sort_values(by=["lemma","count"], ascending=[True,False], inplace=True)

mascpl = df[(df["gender"] == "m") & (df["number"] == "pl")]
mascsg = df[(df["gender"] == "m") & (df["number"] == "sg")]
masc = df[(df["gender"] == "m") & (df["number"] == "")]
fempl = df[(df["gender"] == "f") & (df["number"] == "pl")]
femsg = df[(df["gender"] == "f") & (df["number"] == "sg")]
fem = df[(df["gender"] == "f") & (df["number"] == "")]
pl = df[(df["gender"] == "") & (df["number"] == "pl")]
sg = df[(df["gender"] == "") & (df["number"] == "sg")]

def dedup_lemma(df):
    df["lemma_lag"] = df["lemma"].shift(1)
    df.drop(df[(df["lemma"] == df["lemma_lag"])].index, axis=0, inplace=True)
    df.drop("lemma_lag", axis=1, inplace=True)

dedup_lemma(mascsg)
dedup_lemma(femsg)
dedup_lemma(mascpl)
dedup_lemma(fempl)
dedup_lemma(masc)
dedup_lemma(fem)

sg_mascfem = pd.merge(
    left=mascsg,
    right=femsg,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

pl_mascfem = pd.merge(
    left=mascpl,
    right=fempl,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

na_mascfem = pd.merge(
    left=masc,
    right=fem,
    on="lemma",
    how="outer",
    suffixes=("_masc", "_fem")
)

#%%
def filter_adjs(df, drop_threshold=1000):
    filtered_df = df.dropna(subset=["adj_masc", "adj_fem"], inplace=False)
    filtered_df.drop(
        filtered_df[filtered_df["adj_masc"] == filtered_df["adj_fem"]].index,
        axis=0, inplace=True
    )
    drop_df = filtered_df.drop(
        filtered_df[(filtered_df["count_masc"] < drop_threshold) & 
                (filtered_df["count_fem"] < drop_threshold)].index, 
        axis=0
    )
    return drop_df

#%%
drop_threshold = 1000
sglist = filter_adjs(sg_mascfem, drop_threshold)
pllist = filter_adjs(pl_mascfem, drop_threshold)
nalist = filter_adjs(na_mascfem, drop_threshold)

lemma_list = pd.concat([sglist, pllist, nalist], axis=0)

#%%
lemma_list["pair_total"] = lemma_list["count_masc"] + lemma_list["count_fem"]
lemma_list["pair_p_0"] = lemma_list["count_masc"] / lemma_list["pair_total"]
lemma_list["pair_p_1"] = lemma_list["count_fem"] / lemma_list["pair_total"]

lemma_final = lemma_list[["adj_masc", "adj_fem", "pair_p_0", "pair_p_1"]] # "count_masc", "count_fem", "number_masc"]]
lemma_final.rename(
    {"adj_masc":"lemma_0",
     "adj_fem": "lemma_1"}, 
    axis=1,
    inplace=True
)

#init_list_outpath = os.path.join(DATASETS, "processed/ud_fr_gsd/word_lists/init_adjlist.tsv")
#final_list.to_csv(init_list_outpath, sep="\t")
adj_list_outpath = os.path.join(DATASETS, "processed/fr/word_lists/adj_pair_list.tsv")
lemma_final.to_csv(adj_list_outpath, sep="\t")

#%% concept prob
totals = lemma_list.sum()[["count_masc", "count_fem"]]

totals["total_wout_other"] = totals.sum()
totals["total_incl_other"] = total_count
totals["p_0_wout_other"] = totals["count_masc"] / totals["total_wout_other"]
totals["p_1_wout_other"] = totals["count_fem"] / totals["total_wout_other"]
totals["p_0_incl_other"] = totals["count_masc"] / totals["total_incl_other"]
totals["p_1_incl_other"] = totals["count_fem"] / totals["total_incl_other"]
totals["p_other_incl_other"] = (totals["total_incl_other"] - totals["total_wout_other"]) / totals["total_incl_other"]
totals.rename({"count_masc": "count_0", "count_fem": "count_1"}, inplace=True)

p_concept_outfile = os.path.join(DATASETS, "processed/fr/word_lists/gender_marginals.pkl")
p_concept = totals[["p_0_wout_other", "p_1_wout_other","p_0_incl_other","p_1_incl_other","p_other_incl_other"]]
with open(p_concept_outfile, 'wb') as f:
    pickle.dump(p_concept, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
other_df = pd.DataFrame(other_list)

other_sub = other_df[other_df["count"]>drop_threshold]
other_sub.sort_values(by="count", inplace=True, ascending=False)

#%%
masc_adj_list = lemma_final["lemma_0"].unique()
fem_adj_list = lemma_final["lemma_1"].unique()

other_sub["adj_flag"] = other_sub["word"].apply(lambda x: 1 if ((x in masc_adj_list) or (x in fem_adj_list)) else 0)
other_sub["p_incl_other_unnorm"] = other_sub["count"] / total_count
other_sub["p_incl_other"] = other_sub["p_incl_other_unnorm"] / other_sub["p_incl_other_unnorm"].sum()
other_sub.drop(other_sub[other_sub["adj_flag"]==1].index, axis=0, inplace=True)
#%%
other_list_outpath = os.path.join(DATASETS, "processed/fr/word_lists/other_list.tsv")
other_sub[["word", "p_incl_other"]].to_csv(other_list_outpath, sep="\t")

# %%
