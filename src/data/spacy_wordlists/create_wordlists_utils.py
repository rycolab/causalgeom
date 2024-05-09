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
import pandas as pd

sys.path.append('../../')
#sys.path.append('./src/')

from paths import HF_CACHE, OUT, DATASETS

#%% DATA LOADING
def load_wiki_wordlist(language):
    if language == "en":
        OLD_OUT_FILE = os.path.join(OUT, f"wordlist/{language}_new.pkl")
    else:
        OLD_OUT_FILE = os.path.join(OUT, f"wordlist/{language}_new_0.pkl")
    
    #TODO: replace once these have re-run
    NEW_OUT_FILE = os.path.join(DATASETS, f"{language}/{language}_wiki_wordlist.pkl")

    with open(OLD_OUT_FILE, 'rb') as f:
        token_data = pickle.load(f)
    return token_data

#%% TOKEN LIST FILTER FUNCTIONS
def find_substr(fullstr, substr):
    substr_ind = fullstr.find(substr)
    if substr_ind > -1:
        substr_ind = substr_ind + len(substr)
        return fullstr[substr_ind:substr_ind+4]
    else:
        return ""

def get_lemma(lemmadict):
    lowerlemmadict = {}
    for k, v in lemmadict.items():
        lowerv = lemmadict.get(k.lower(), 0)
        lowerlemmadict[k] = lowerv + v
    maxkey = max(lowerlemmadict, key=lowerlemmadict.get)
    return maxkey, lowerlemmadict[maxkey]

#%% LEMMA LIST FUNCTIONS
def dedup_lemma(df):
    df["lemma_lag"] = df["lemma"].shift(1)
    df.drop(df[(df["lemma"] == df["lemma_lag"])].index, axis=0, inplace=True)
    df.drop("lemma_lag", axis=1, inplace=True)

# %% OTHER TOKEN LIST FUNCTIONS
def filter_other_list(other_list, l0_list, l1_list, drop_threshold):
    other_df = pd.DataFrame(other_list)

    other_sub = other_df[other_df["count"]>drop_threshold]
    other_sub.sort_values(by="count", inplace=True, ascending=True)

    other_sub["lemma_flag"] = other_sub["word"].apply(
        lambda x: 1 if ((x in l0_list) or (x in l1_list)) else 0
    )
    #other_sub["p_incl_other_unnorm"] = other_sub["count"] / total_count
    #other_sub["p_incl_other"] = other_sub["p_incl_other_unnorm"] / other_sub["p_incl_other_unnorm"].sum()
    other_sub.drop(
        other_sub[other_sub["lemma_flag"]==1].index, 
        axis=0, inplace=True
    )
    return other_sub[["word"]]#, "p_incl_other"]]