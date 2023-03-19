#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import pandas as pd

from conllu import parse_incr, parse_tree_incr
from inflecteur import inflecteur

sys.path.append('../../')
#sys.path.append('./src/')

from paths import UD_FRENCH_GSD, DATASETS


#%%
SPLIT = "test"
INPUT_FILE = os.path.join(UD_FRENCH_GSD, f"fr_gsd-ud-{SPLIT}.conllu")
OUTPUT_FILE = os.path.join(DATASETS, f"preprocessed/ud/fr/gsd/{SPLIT}.pkl")

#%%
def get_feature(feats, feature_name, default_val=None):
    try:
        return feats.get(feature_name, default_val)
    except AttributeError:
        return default_val

def process_child(metadata_text, child, sent_index, level):
    if child.token["upos"] == "NOUN":
        noun = child.token["form"].lower()
        noun_index = child.token["id"] - 1
        feats = child.token.get("feats", {})
        gender = get_feature(feats, "Gender")
        number = get_feature(feats, "Number")
        typo = get_feature(feats, "Typo")
        tense = get_feature(feats, "Tense")
        foreign = get_feature(feats, "Foreign")
        adjs = []
        for child1 in child.children:
            if child1.token["upos"] == "ADJ":
                adj_feats = child1.token.get("feats", {})
                adj_gender = get_feature(adj_feats, "Gender")
                adj_number = get_feature(adj_feats, "Number")
                adj_numtype = get_feature(adj_feats, "NumType")
                adj_typo = get_feature(adj_feats, "Typo")
                #adj_tense = get_feature(adj_feats, "Tense")
                #adj_foreign = get_feature(adj_feats, "Foreign")
                adj_prontype = get_feature(adj_feats, "PronType")
                adj_dict = dict(
                    adj=child1.token["form"].lower(), 
                    index=child1.token["id"] - 1,
                    deprel=child1.token["deprel"], 
                    feats=adj_feats,
                    gender=adj_gender, 
                    number=adj_number, 
                    numtype=adj_numtype, 
                    typo=adj_typo,
                    prontype=adj_prontype
                )
                adjs.append(adj_dict)
        noun_dict = dict(
            sent_index=sent_index,
            level=level,
            noun=noun,
            noun_index=noun_index,
            feats=feats,
            gender=gender,
            number=number,
            typo=typo,
            tense=tense,
            foreign=foreign,
            adjs=adjs,
            text=metadata_text.lower()
        )
        return noun_dict
    else:
        return None

def collect_noun_det_pairs(metadata_text, tokentree, sent_index, level):
    ndps = []
    for child in tokentree.children:
    #child0 = tester.children[0]
        res = process_child(metadata_text, child, sent_index, level)
        if res is not None:
            ndps += [res] + collect_noun_det_pairs(metadata_text, child, sent_index, level+1)
        else:
            ndps += collect_noun_det_pairs(metadata_text, child, sent_index, level+1)
    return ndps
    
#%%
ndps = []
tts = []
tls = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    i = 0
    for treelist in parse_tree_incr(f):
        tts.append(treelist)
        metadata_text = treelist.metadata["text"]
        ndps += collect_noun_det_pairs(metadata_text, treelist, i, 0)
        i += 1

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for tokenlist in parse_incr(f):
        tls.append(tokenlist)


# %%
df = pd.DataFrame(ndps)

#%%
# adj drop
df["adjlen"] = df["adjs"].apply(len)

df.drop(df[(df["adjlen"]!=1)].index, inplace=True)

#%% bad noun drops 
# noun feature investigation
df["feats"].apply(lambda x: x.keys() if x is not None else []).value_counts()

df.drop(df[(df["gender"].isnull())].index, inplace=True)
df.drop(df[(df["typo"]=="Yes")].index, inplace=True)
df.drop(df[(df["foreign"]=="Yes")].index, inplace=True)

df.drop(["typo", "tense", "foreign", "feats"],axis=1, inplace=True)

#%% Creating ADJ df
adj_df = pd.DataFrame(list(df["adjs"].apply(lambda x: x[0])))
adj_df.columns = ["adj_" + x for x in adj_df.columns]
df.drop(["adjs", "adjlen"],axis=1, inplace=True)
df.reset_index(drop=True,inplace=True)
adj_df.reset_index(drop=True,inplace=True)

full_df = pd.concat((df, adj_df), axis=1, ignore_index=False)

#%% Merging Adjs and Adj Filtering
# check adjs feats
full_df["adj_feats"].apply(lambda x: x.keys() if x is not None else []).value_counts()

# drop bad adjs
full_df.drop(full_df[(full_df["adj_deprel"]!="amod")].index, inplace=True)
full_df.drop(full_df[(full_df["adj_gender"].isnull())].index, inplace=True)
full_df.drop(full_df[(full_df["adj_number"].isnull())].index, inplace=True)
full_df.drop(full_df[(full_df["adj_numtype"].notnull())].index, inplace=True)
full_df.drop(full_df[(full_df["adj_typo"]=="Yes")].index, inplace=True)
full_df.drop(full_df[(full_df["adj_prontype"]=="Ind")].index, inplace=True)
full_df["adj"] = full_df["adj_adj"].apply(lambda x: x.lower())
full_df.drop(["adj_adj", "adj_deprel", "adj_numtype", "adj_typo", "adj_prontype"], axis=1, inplace=True)

full_df.drop(
    full_df[(full_df["number"] != full_df["adj_number"]) | 
            (full_df["gender"] != full_df["adj_gender"])].index,
    axis=0, inplace=True
)


#%% mislabeled adjs -- DID NOT FINISH THIS FILTER
"""
adjs = full_df.groupby(["adj", "adj_gender", "adj_number"])["sent_index"].count().reset_index()
adjs.columns = ["adj", "adj_gender", "adj_number", "cnt"]
adjs.sort_values(by = ["adj", "cnt"], ascending=[True, False])

adjs["lag_adj"] = adjs["adj"].shift(1)
adjs["lag_cnt"] = adjs["cnt"].shift(1)
adjs["lag_adj"].fillna(value="", inplace=True)
adjs["cnt"] = adjs["cnt"].astype(int)
adjs["lag_cnt"] = adjs["lag_cnt"].fillna(value=0).astype(int)

def flag_dups(adj, lag_adj, cnt, lag_cnt):
    if (adj == lag_adj) and (cnt < lag_cnt):
        return 1
    elif (adj == lag_adj) and (cnt == lag_cnt):
        return 2
    else:
        return 0 

adjs["drop_flag"] = adjs.apply(lambda x: flag_dups(x.adj, x.lag_adj, int(x.cnt), int(x.lag_cnt)), axis=1)
"""
#%%
INFL = inflecteur()
INFL.load_dict()

def inflect_adj(adj, gender): 
    if gender=="Masc":
        return INFL.inflect_sentence(adj, gender="f")
    elif gender=="Fem":
        return INFL.inflect_sentence(adj, gender="m")
    else:
        return None

def inflect_adj_handler(adj, gender):
    try:
        return inflect_adj(adj, gender)
    except IndexError:
        print(f"Index error, bad string: {adj}")
        return None

all_adjs = full_df[["adj", "adj_gender"]].drop_duplicates()
all_adjs["adj_gender_foil"] = all_adjs.apply(lambda x: inflect_adj_handler(x.adj, x.adj_gender) , axis=1)

#%% Cleaning foils
good_adjs = all_adjs.drop(
    all_adjs[(all_adjs["adj"] == all_adjs["adj_gender_foil"])].index
)

#%% Merging foil back on
full_df_foil = pd.merge(
    left=full_df,
    right=good_adjs[["adj", "adj_gender", "adj_gender_foil"]],
    on=["adj", "adj_gender"],
    how="outer"
)
full_df_foil.drop(
    full_df_foil[full_df_foil["adj_gender_foil"].isnull()].index, inplace=True
)

#%% Final processing
def flag_masked_sentence(sentence, adj, adj_index):
    try:
        find_adj_ind = sentence.split(" ").index(adj)
    except ValueError:
        return 0
    else:
        if find_adj_ind == adj_index:
            return 1
        else:
            return 2

def get_masked_sentence(sentence, adj):
    try:
        split_sentence = sentence.split(" ")
        find_adj_ind = split_sentence.index(adj)
    except ValueError:
        if sentence.count(adj) == 1:
            return sentence.replace(adj, "[MASK]", 1)
        else:
            return None
    else:
        split_sentence[find_adj_ind] = "[MASK]"
        return " ".join(split_sentence)

full_df_foil["masked_check"] = full_df_foil.apply(
    lambda x: flag_masked_sentence(x.text, x.adj, x.adj_index), axis=1
)
full_df_foil["masked"] = full_df_foil.apply(
    lambda x: get_masked_sentence(x.text, x.adj), axis=1
)
full_df_foil.drop(full_df_foil[full_df_foil["masked"].isnull()].index, inplace=True)

full_df_foil["ar_flag"] = (full_df_foil["noun_index"] < full_df_foil["adj_index"])


#%%
final_df = full_df_foil[
    ["level", "noun", "gender", "number", "adj", "adj_gender_foil", 
    "text", "masked", "ar_flag"]
]
final_df.to_pickle(OUTPUT_FILE)

