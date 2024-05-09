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


# %% Number Specific Functions
def get_number(tag):
    if tag == "VBP":
        return "pl"
    elif tag == "VBZ":
        return "sg"
    else:
        raise ValueError("Tag not supported")

def process_token_data_number(token_data):
    verb_dict = {}
    verb_list = []
    other_list = []
    total_count = 0
    for k, v in token_data.items():
        token_tags = dict(sorted(v["token_tag"].items(), reverse=True))
        token_count = v["count"]
        for kp, vp in token_tags.items():
            if kp.strip() in ["VBP", "VBZ"] and (vp / token_count) > .01 and vp > 100:
                #gennumstr, gender, number = get_gender_number(v["token_morph"])
                number = get_number(kp.strip())
                lemma, lemmacount = get_lemma(v["token_lemma"])
                res = dict(
                    verb=k,
                    lemma=lemma,
                    count=vp, 
                    number=number, 
                    number_ratio=vp/token_count,
                    lemma_ratio=lemmacount/vp
                )
                verb_dict[k] = res
                verb_list.append(res)
                break
            else:
                continue
        other_list.append(
            dict(
                word=k,
                count=token_count
            )
        )
        total_count+=token_count
    return verb_dict, verb_list, other_list, total_count

#%%
def get_sgpl_df(verb_list):
    df = pd.DataFrame(verb_list)
    df.sort_values(by=["lemma","count"], ascending=[True,False], inplace=True)
    pl = df[df["number"] == "pl"]
    sg = df[df["number"] == "sg"]

    dedup_lemma(pl)
    dedup_lemma(sg)

    sgpl = pd.merge(
        left=sg,
        right=pl,
        on="lemma",
        how="outer",
        suffixes=("_sg", "_pl")
    )
    return sgpl

#%%
def filter_merge(df, drop_threshold=1000):
    filtered_df = df.dropna(subset=["verb_sg", "verb_pl"], inplace=False)

    drop_df = filtered_df.drop(
        filtered_df[(filtered_df["count_sg"] < drop_threshold) & 
                (filtered_df["count_pl"] < drop_threshold)].index, 
        axis=0
    )
    return drop_df

def format_verb_list(sgpl_filtered):
    #sgpl_filtered["pair_total"] = sgpl_filtered["count_sg"] + sgpl_filtered["count_pl"]
    #sgpl_filtered["pair_p_0"] = sgpl_filtered["count_sg"] / sgpl_filtered["pair_total"]
    #sgpl_filtered["pair_p_1"] = sgpl_filtered["count_pl"] / sgpl_filtered["pair_total"]

    lemma_final = sgpl_filtered[["verb_sg", "verb_pl"]]#, "pair_p_0", "pair_p_1"]] # "count_masc", "count_fem", "number_masc"]]
    lemma_final.rename(
        {"verb_sg":"lemma_0",
        "verb_pl": "lemma_1"}, 
        axis=1,
        inplace=True
    )
    return lemma_final

#%% concept prob
#totals = sgpl_filtered.sum()[["count_sg", "count_pl"]]
#totals["total_wout_other"] = totals.sum()
#totals["total_incl_other"] = total_count
#totals["p_0_wout_other"] = totals["count_sg"] / totals["total_wout_other"]
#totals["p_1_wout_other"] = totals["count_pl"] / totals["total_wout_other"]
#totals["p_0_incl_other"] = totals["count_sg"] / totals["total_incl_other"]
#totals["p_1_incl_other"] = totals["count_pl"] / totals["total_incl_other"]
#totals["p_other_incl_other"] = (totals["total_incl_other"] - totals["total_wout_other"]) / totals["total_incl_other"]
#totals.rename({"count_sg": "count_0", "count_pl": "count_1"}, inplace=True)

#p_concept_outfile = os.path.join(DATASETS, "processed/en/word_lists/number_marginals.pkl")
#p_concept = totals[
#    ["p_0_wout_other", "p_1_wout_other","p_0_incl_other","p_1_incl_other","p_other_incl_other"]
#]
#with open(p_concept_outfile, 'wb') as f:
#    pickle.dump(p_concept, f, protocol=pickle.HIGHEST_PROTOCOL)



#%%
# PARAMS
language = "en"
drop_threshold = 1000

# Loading Data
token_data = load_wiki_wordlist(language)

# Creating Export Directory
EXPORT_DIR = os.path.join(DATASETS, f"processed/{language}/number_word_lists")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Processing Tagged Wikipedia Words
verb_dict, verb_list, other_list, total_count = process_token_data_number(token_data)

# Creating verb lemma pairs
sgpl = get_sgpl_df(verb_list)
sgpl_filtered = filter_merge(sgpl, drop_threshold)
sgpl_formatted = format_verb_list(sgpl_filtered)

verb_list_outpath = os.path.join(EXPORT_DIR, "verb_pair_list.tsv")
sgpl_formatted.to_csv(verb_list_outpath, sep="\t")

# Filtering other words to exclude lemma words
sg_verb_list = sgpl_formatted["lemma_0"].unique().tolist()
pl_verb_list = sgpl_formatted["lemma_1"].unique().tolist()

filtered_other_list = filter_other_list(
    other_list, sg_verb_list, pl_verb_list, drop_threshold
)
other_list_outpath = os.path.join(EXPORT_DIR, "number_other_list.tsv")
filtered_other_list.to_csv(other_list_outpath, sep="\t")
