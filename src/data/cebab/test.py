#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import datasets
import numpy as np

sys.path.append('../..')
#sys.path.append('./src/')
from utils.auth_utils import load_auth_token
from data.cebab.data_utils import preprocess_hf_dataset
from paths import HF_CACHE

#%%
# column names
OVERALL_LABEL = 'review_majority'
DESCRIPTION = 'description'
TREATMENTS = ['food', 'ambiance', 'service', 'noise']
NO_MAJORITY = 'no majority'
ID_COL = 'id'

# possible concept values
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
UNKNOWN = 'unknown'

# task names
OPENTABLE_BINARY = 'opentable_binary'
OPENTABLE_TERNARY = 'opentable_ternary'
OPENTABLE_5_WAY = 'opentable_5_way'


#%% data
TASK_NAME = "opentable_binary"

cebab = datasets.load_dataset('CEBaB/CEBaB', token=load_auth_token(), cache_dir=HF_CACHE)
if TASK_NAME == OPENTABLE_BINARY:
    NUM_CLASSES = 2
elif TASK_NAME == OPENTABLE_TERNARY:
    NUM_CLASSES = 3
elif TASK_NAME == OPENTABLE_5_WAY:
    NUM_CLASSES = 5
else:
    raise ValueError(f'Unsupported task \"{TASK_NAME}\"')

#%%
#args.str_num_classes = f'{NUM_CLASSES}-class'
DATASET_TYPE = f'{NUM_CLASSES}-way'
train, dev, test = preprocess_hf_dataset(
    cebab, 
    one_example_per_world=True, 
    verbose=1,
    dataset_type=DATASET_TYPE
)
# %%
def add_suffix(text, suffix):
    if text.strip().endswith("."):
        return text + " " + suffix
    else:
        return text + ". " + suffix

def format_data(df, suffix):
    formatted_df = df[
        ['id', 'description', 'review_majority', 'food_aspect_majority',
        'ambiance_aspect_majority', 'service_aspect_majority',
        'noise_aspect_majority']
    ]
    formatted_df.columns = [
        'id', 'text', 'overall_sentiment', 'food_sentiment',
        'ambiance_sentiment', 'service_sentiment',
        'noise_sentiment'
    ]

    formatted_df["pre_tgt_text"] = formatted_df["text"].apply(lambda x: add_suffix(x, test_suffix))
    formatted_df["fact"] = np.where(formatted_df["overall_sentiment"] == 1, " up", " down")
    formatted_df["foil"] = np.where(formatted_df["overall_sentiment"] == 1, " down", " up")
    formatted_df["fact_text"] = formatted_df["pre_tgt_text"] + formatted_df["fact"]
    formatted_df["foil_text"] = formatted_df["pre_tgt_text"] + formatted_df["foil"]

    final_df = formatted_df[["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", "overall_sentiment"]]
    final_df.columns = ["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", "tgt_label"]

    return final_df

test_suffix = "Thumbs"
final_train, final_dev, final_test = format_data(train, test_suffix), format_data(dev, test_suffix), format_data(test, test_suffix)

#%%
from paths import DATASETS
import pickle 

CEBaB_PATH = os.path.join(DATASETS, "preprocessed/CEBaB/test")
splits = [(final_train, "train"),(final_dev, "dev"),(final_test, "test")]
for data, split in splits:
    fpath = os.path.join(CEBaB_PATH, f"CEBaB_{split}.pkl")
    with open(fpath, "wb") as f:
        pickle.dump(data, f)



# %%
#from utils.dataset_loaders import load_preprocessed_dataset

#linzen_data = load_preprocessed_dataset("linzen", "gpt2-large")



    

# %%
