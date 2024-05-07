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

sys.path.append('../../')
#sys.path.append('./src/')

from paths import HF_CACHE, OUT, DATASETS
from data.spacy_wordlists.create_wordlists_utils import load_wiki_wordlist,\
    find_substr, get_lemma, dedup_lemma, filter_other_list


# %% Gender Specific Functions
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
    
def process_token_data_gender(token_data):
    adj_dict = {}
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
                adj_dict[k] = res
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
    return adj_dict, adj_list, other_list, total_count

def get_unique_gender_number_vals(adj_dict):
    """ for diagnostic purposes """
    gender_vals = set()
    number_vals = set()
    for k, v in adj_dict.items():
        gender_vals.update([v["gender"]])
        number_vals.update([v["number"]])
    return gender_vals, number_vals

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

def create_paired_adj_list(adj_list, drop_threshold):
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

    sglist = filter_adjs(sg_mascfem, drop_threshold)
    pllist = filter_adjs(pl_mascfem, drop_threshold)
    nalist = filter_adjs(na_mascfem, drop_threshold)

    lemma_list = pd.concat([sglist, pllist, nalist], axis=0)

    return lemma_list

def format_adj_list(lemma_list):
    #lemma_list["pair_total"] = lemma_list["count_masc"] + lemma_list["count_fem"]
    #lemma_list["pair_p_0"] = lemma_list["count_masc"] / lemma_list["pair_total"]
    #lemma_list["pair_p_1"] = lemma_list["count_fem"] / lemma_list["pair_total"]

    lemma_final = lemma_list[["adj_masc", "adj_fem"]]#, "pair_p_0", "pair_p_1"]] # "count_masc", "count_fem", "number_masc"]]
    lemma_final.rename(
        {"adj_masc":"lemma_0",
        "adj_fem": "lemma_1"}, 
        axis=1,
        inplace=True
    )
    return lemma_final

#%%
# PARAMS
language = "fr"
drop_threshold = 1000

# Loading Data
token_data = load_wiki_wordlist(language)

# Creating Export Directory
EXPORT_DIR = os.path.join(DATASETS, f"processed/{language}/number_word_lists")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Processing Tagged Wikipedia Words
adj_dict, adj_list, other_list, total_count = process_token_data_gender(token_data)

# Creating adj lemma pairs
paired_adjs = create_paired_adj_list(adj_list, drop_threshold)
formatted_paired_adjs = format_adj_list(paired_adjs)

adj_list_outpath = os.path.join(EXPORT_DIR, "adj_pair_list.tsv")
formatted_paired_adjs.to_csv(adj_list_outpath, sep="\t")

# Filtering other words to exclude lemma words
l0_word_list = formatted_paired_adjs["lemma_0"].unique().tolist()
l1_word_list = formatted_paired_adjs["lemma_1"].unique().tolist()

filtered_other_list = filter_other_list(
    other_list, l0_word_list, l1_word_list, drop_threshold
)
other_list_outpath = os.path.join(EXPORT_DIR, "gender_other_list.tsv")
filtered_other_list.to_csv(other_list_outpath, sep="\t")

#%% concept prob
#totals = lemma_list.sum()[["count_masc", "count_fem"]]

#totals["total_wout_other"] = totals.sum()
#totals["total_incl_other"] = total_count
#totals["p_0_wout_other"] = totals["count_masc"] / totals["total_wout_other"]
#totals["p_1_wout_other"] = totals["count_fem"] / totals["total_wout_other"]
#totals["p_0_incl_other"] = totals["count_masc"] / totals["total_incl_other"]
#totals["p_1_incl_other"] = totals["count_fem"] / totals["total_incl_other"]
#totals["p_other_incl_other"] = (totals["total_incl_other"] - totals["total_wout_other"]) / totals["total_incl_other"]
#totals.rename({"count_masc": "count_0", "count_fem": "count_1"}, inplace=True)

#p_concept_outfile = os.path.join(DATASETS, "processed/fr/word_lists/gender_marginals.pkl")
#p_concept = totals[["p_0_wout_other", "p_1_wout_other","p_0_incl_other","p_1_incl_other","p_other_incl_other"]]
#with open(p_concept_outfile, 'wb') as f:
#    pickle.dump(p_concept, f, protocol=pickle.HIGHEST_PROTOCOL)

