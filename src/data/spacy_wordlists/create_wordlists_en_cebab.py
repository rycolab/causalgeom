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

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
CEBAB_FOOD_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/food_adjs.tsv")
CEBAB_AMBIANCE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/ambiance_adjs.tsv")
CEBAB_NOISE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/noise_adjs.tsv")
CEBAB_SERVICE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/service_adjs.tsv")

#%%#################
# Paths            #
####################
def get_wordlist_paths(concept):
    if concept == "food":
        return CEBAB_FOOD_ADJ_LIST_PATH
    elif concept == "ambiance":
        return CEBAB_AMBIANCE_ADJ_LIST_PATH
    elif concept == "noise":
        return CEBAB_NOISE_ADJ_LIST_PATH
    elif concept == "service":
        return CEBAB_SERVICE_ADJ_LIST_PATH
    else:
        raise NotImplementedError("Invalid dataset name")

#%% 
def process_token_data(token_data):
    token_list = []
    total_count = 0
    for k, v in token_data.items():
        token_count = v["count"]
        token_list.append(
            dict(
                word=k,
                count=token_count
            )
        )
        total_count+=token_count
    return token_list, total_count

#%%
def create_and_export_other_list(all_words_list, concept, 
    drop_threshold, export_dir):
    concept_word_list_path = get_wordlist_paths(concept)
    concept_word_list = pd.read_csv(
        concept_word_list_path, index_col=0, sep="\t"
    )

    # Filtering other words to exclude lemma words
    l0_word_list = concept_word_list["lemma_0"].unique().tolist()
    l1_word_list = concept_word_list["lemma_1"].unique().tolist()

    filtered_other_list = filter_other_list(
        all_words_list, l0_word_list, l1_word_list, drop_threshold
    )
    logging.info(f"Number of other words for concept {concept}: {filtered_other_list.shape[0]}")
    other_list_outpath = os.path.join(export_dir, f"{concept}_other_list.tsv")
    filtered_other_list.to_csv(other_list_outpath, sep="\t")
    logging.info(f"Other word list for {concept} exported to {other_list_outpath}")

#%%
# PARAMS
language = "en"
drop_threshold = 1000

# Loading Data
token_data = load_wiki_wordlist(language)

# Creating Export Directory
EXPORT_DIR = os.path.join(DATASETS, f"processed/{language}/CEBaB_word_lists")
os.makedirs(EXPORT_DIR, exist_ok=True)

all_words_list, total_count = process_token_data(token_data)

create_and_export_other_list(all_words_list, "ambiance", drop_threshold, EXPORT_DIR)
create_and_export_other_list(all_words_list, "food", drop_threshold, EXPORT_DIR)
create_and_export_other_list(all_words_list, "noise", drop_threshold, EXPORT_DIR)
create_and_export_other_list(all_words_list, "service", drop_threshold, EXPORT_DIR)

# %%
