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

#sys.path.append('../../')
sys.path.append('./src/')

from paths import DATASETS, UD_FRENCH_GSD, UD_FRENCH_ParTUT, UD_FRENCH_Rhapsodie

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "-dataset", 
        type=str,
        choices=["ud_fr_gsd", "ud_fr_partut", "ud_fr_rhapsodie"],
        help="Dataset to preprocess"
    )
    argparser.add_argument(
        "-split",
        type=str,
        choices=["train", "dev", "test"],
        default=None,
        help="While split to preprocess"
    )
    return argparser.parse_args()


args = get_args()
logging.info(args)
DATASET = args.dataset
SPLIT = args.split
#DATASET = "ud_fr_partut"
#SPLIT = "train"

def get_input_file(dataset, split):
    if dataset == "ud_fr_gsd":
        input_file = os.path.join(UD_FRENCH_GSD, f"fr_gsd-ud-{split}.conllu")
    elif dataset == "ud_fr_partut":
        input_file = os.path.join(UD_FRENCH_ParTUT, f"fr_partut-ud-{split}.conllu")
    elif dataset == "ud_fr_rhapsodie":
        input_file = os.path.join(UD_FRENCH_Rhapsodie, f"fr_rhapsodie-ud-{split}.conllu")
    else:
        raise ValueError("Unsupported dataset")
    return input_file

INPUT_FILE = get_input_file(DATASET, SPLIT)
OUTPUT_FILE = os.path.join(DATASETS, f"preprocessed/{DATASET}/{SPLIT}.pkl")
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
logging.info(f"Preprocessing UD French data split: {SPLIT}")

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
df.drop(df[(df["adjlen"]==0)].index, inplace=True)

df["first_adj"] = df["adjs"].apply(lambda x: x[0])

#%% bad noun drops 
# noun feature investigation
df["feats"].apply(lambda x: x.keys() if x is not None else []).value_counts()

df.drop(df[(df["gender"].isnull())].index, inplace=True)
df.drop(df[(df["typo"]=="Yes")].index, inplace=True)
df.drop(df[(df["foreign"]=="Yes")].index, inplace=True)

df.drop(["typo", "tense", "foreign", "feats"],axis=1, inplace=True)

#%% Creating ADJ df
adj_df = pd.DataFrame(list(df["first_adj"]))
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

#%% ar and masked sentence processing
def flag_valid_sentence(sentence, adj, adj_index):
    try:
        list_adj_ind = sentence.split(" ").index(adj)
    except ValueError:
        find_adj_ind = sentence.find(adj)
        if find_adj_ind == -1:
            return 0
        elif sentence.count(adj) == 1:
            return 3
        elif sentence.count(adj) > 1:
            return 4
        else:
            return 5
    else:
        if list_adj_ind == adj_index:
            return 1
        else:
            return 2

def get_masked_sentence(sentence, adj, valid_flag):
    if sentence.count(adj) == 1 and valid_flag in [1,2,3]:
        return sentence.replace(adj, "<mask>", 1)
    else:
        return None

def get_ar_sentence(sentence, adj, valid_flag):
    if sentence.count(adj) == 1 and valid_flag in [1,2,3]:
        return sentence[:sentence.find(adj)]
    else:
        return None

full_df_foil["valid_flag"] = full_df_foil.apply(
    lambda x: flag_valid_sentence(x.text, x.adj, x.adj_index), axis=1
)
logging.info(f"Valid flag breakdown: \n {full_df_foil['valid_flag'].value_counts()}")

full_df_foil["masked"] = full_df_foil.apply(
    lambda x: get_masked_sentence(x.text, x.adj, x.valid_flag), axis=1
)

full_df_foil["ar_flag"] = (full_df_foil["noun_index"] < full_df_foil["adj_index"])
full_df_foil["pre_tgt_text"] = full_df_foil.apply(
    lambda x: get_ar_sentence(x.text, x.adj, x.valid_flag), axis=1
)

full_df_foil["fact_text"] = full_df_foil["pre_tgt_text"] + full_df_foil["adj"]
full_df_foil["foil_text"] = full_df_foil["pre_tgt_text"] + full_df_foil["adj_gender_foil"]

#%%
assert full_df_foil[full_df_foil["masked"].isnull()].shape == \
    full_df_foil[full_df_foil["pre_tgt_text"].isnull()].shape, \
        "Number of sentences with two instances of tgt have to match"

full_df_foil.drop(
    full_df_foil[full_df_foil["masked"].isnull()].index, inplace=True)

final_df = full_df_foil[
    ["level", "noun", "gender", "number", "adj", "adj_gender_foil", 
    "text", "masked", "ar_flag", "pre_tgt_text", "fact_text", "foil_text"]
]
final_df.to_pickle(OUTPUT_FILE)

logging.info(f"Exported to {OUTPUT_FILE}, number of obs: {final_df.shape[0]}")
# %%
