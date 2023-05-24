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

from paths import HF_CACHE, OUT, DATASETS

#%% UD FR DATA FOR REFERENCE
#DATA_FILE = os.path.join(DATASETS, "preprocessed/ud_fr_gsd/train.pkl")
#with open(DATA_FILE, 'rb') as f:      
#    sent_data = pickle.load(f)

#%%
language = "en"
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

def get_number(tag):
    if tag == "VBP":
        return "pl"
    elif tag == "VBZ":
        return "sg"
    else:
        raise ValueError("Tag not supported")

def get_lemma(lemmadict):
    lowerlemmadict = {}
    for k, v in lemmadict.items():
        lowerv = lemmadict.get(k.lower(), 0)
        lowerlemmadict[k] = lowerv + v
    maxkey = max(lowerlemmadict, key=lowerlemmadict.get)
    return maxkey, lowerlemmadict[maxkey]

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


#%% Diagnostics
df = pd.DataFrame(verb_list)
df.hist("number_ratio")
df.sort_values("number_ratio")
df.sort_values(by=["lemma","count"], ascending=[True,False], inplace=True)
pl = df[df["number"] == "pl"]
sg = df[df["number"] == "sg"]


def dedup_lemma(df):
    df["lemma_lag"] = df["lemma"].shift(1)
    df.drop(df[(df["lemma"] == df["lemma_lag"])].index, axis=0, inplace=True)
    df.drop("lemma_lag", axis=1, inplace=True)

dedup_lemma(pl)
dedup_lemma(sg)

sgpl = pd.merge(
    left=sg,
    right=pl,
    on="lemma",
    how="outer",
    suffixes=("_sg", "_pl")
)

#%%
# TODO: 
# - once enough data gathered, clean up said lists
# - handle dup lists manually
# - take all other tokens and filter out the ones that are in the tgt list
# done :)

#%%
def filter_merge(df, drop_threshold=1000):
    filtered_df = df.dropna(subset=["verb_sg", "verb_pl"], inplace=False)
    drop_df = filtered_df.drop(
        filtered_df[(filtered_df["count_sg"] < drop_threshold) & 
                (filtered_df["count_pl"] < drop_threshold)].index, 
        axis=0
    )
    return drop_df

#%%
drop_threshold = 1000
sgpl_filtered = filter_merge(sgpl, drop_threshold)

sgpl_filtered["total_count"] = total_count
sgpl_filtered["p_0"] = sgpl_filtered["count_sg"] / sgpl_filtered["total_count"]
sgpl_filtered["p_1"] = sgpl_filtered["count_pl"] / sgpl_filtered["total_count"]
#TODO: fix this tomorrow, not sure what to make of this.
lemma_final = sgpl_filtered[["verb_sg", "verb_pl", "p_0", "p_1"]] # "count_masc", "count_fem", "number_masc"]]
lemma_final.rename(
    {"verb_sg":"lemma_0",
     "verb_pl": "lemma_1"}, 
    axis=1,
    inplace=True
)

# %%
#init_list_outpath = os.path.join(DATASETS, "processed/ud_fr_gsd/word_lists/init_adjlist.tsv")
#final_list.to_csv(init_list_outpath, sep="\t")
adj_list_outpath = os.path.join(DATASETS, "processed/fr/word_lists/adj_list.tsv")
lemma_final.to_csv(adj_list_outpath, sep="\t")

#%% concept prob
totals = lemma_list.sum()[["count_masc", "count_fem"]]
totals["total"] = totals.sum()
totals["p_0"] = totals["count_masc"] / totals["total"]
totals["p_1"] = totals["count_fem"] / totals["total"]

p_concept_outfile = os.path.join(DATASETS, "processed/fr/word_lists/p_fr_gender.pkl")
p_concept = totals[["p_0", "p_1"]]
with open(p_concept_outfile, 'wb') as f:
    pickle.dump(p_concept, f, protocol=pickle.HIGHEST_PROTOCOL)

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
