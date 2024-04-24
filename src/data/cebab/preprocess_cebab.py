#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import datasets
import numpy as np
import pickle 

#sys.path.append('../..')
sys.path.append('./src/')

from utils.auth_utils import load_auth_token
from data.cebab.data_utils import preprocess_hf_dataset
from paths import HF_CACHE, DATASETS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


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


#%%
def get_args():
    argparser = argparse.ArgumentParser(description='CEBaB preprocessing args')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["food", "ambiance", "service", "noise"],
        help="Which concept to create dataset for"
    )
    return argparser.parse_args()

args = get_args()
logging.info(args)
    
CONCEPT = args.concept

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

#%% ORIGINAL SETUP HERE 
#args.str_num_classes = f'{NUM_CLASSES}-class'
DATASET_TYPE = f'{NUM_CLASSES}-way'
train, dev, test = preprocess_hf_dataset(
    cebab, 
    one_example_per_world=True, 
    verbose=1,
    dataset_type=DATASET_TYPE
)

# %%
def preprocess_df(df):
    formatted_df = df[
        ['id', 'description', 'review_majority', 'food_aspect_majority',
        'ambiance_aspect_majority', 'service_aspect_majority',
        'noise_aspect_majority']
    ]
    formatted_df.columns = [
        'id', 'pre_tgt_text', 'overall_sentiment', 'food_sentiment',
        'ambiance_sentiment', 'service_sentiment',
        'noise_sentiment'
    ]
    formatted_df["fact"] = "" 
    formatted_df["foil"] = "" 
    formatted_df["fact_text"] = formatted_df["pre_tgt_text"] + formatted_df["fact"]
    formatted_df["foil_text"] = formatted_df["pre_tgt_text"] + formatted_df["foil"]
    return formatted_df

def create_concept_df(df, concept):
    formatted_df = preprocess_df(df)
    concept_df = formatted_df[formatted_df[f"{CONCEPT}_sentiment"].isin([POSITIVE, NEGATIVE])]
    concept_df["tgt_label"] = np.where(concept_df[f"{CONCEPT}_sentiment"] == POSITIVE, 1, 0)

    return concept_df[["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", "tgt_label"]]


#%%
final_train, final_dev, final_test = create_concept_df(train, CONCEPT), create_concept_df(dev, CONCEPT), create_concept_df(test, CONCEPT)


#%%
CEBaB_PATH = os.path.join(DATASETS, f"preprocessed/CEBaB/{CONCEPT}")
os.makedirs(CEBaB_PATH, exist_ok=True)
splits = [(final_train, "train"),(final_dev, "dev"),(final_test, "test")]
for data, split in splits:
    logging.info(f"{split} set label distribution: \n "
                f"{data['tgt_label'].value_counts()}")
    fpath = os.path.join(CEBaB_PATH, f"CEBaB_{CONCEPT}_{split}.pkl")
    with open(fpath, "wb") as f:
        pickle.dump(data, f)

logging.info(f"Successfully exported {CONCEPT} datasets to: {CEBaB_PATH}")

#%%
#def add_suffix(text, suffix):
#    if text.strip().endswith("."):
#        return text + " " + suffix
#    else:
#        return text + ". " + suffix

#FOOD_PROMPTS = ['Food tasted ', 'Cuisine proved ', 'Meal was ', 
#    'Dishes were ', 'The cuisine was ', 'The dishes were ', 'The meal was ']

#formatted_df["pre_tgt_text"] = formatted_df["text"].apply(lambda x: add_suffix(x, test_suffix))
#formatted_df["fact"] = None #np.where(formatted_df["overall_sentiment"] == 1, " up", " down")
#formatted_df["foil"] = None #np.where(formatted_df["overall_sentiment"] == 1, " down", " up")
#formatted_df["fact_text"] = None #formatted_df["pre_tgt_text"] + formatted_df["fact"]
#formatted_df["foil_text"] = None #formatted_df["pre_tgt_text"] + formatted_df["foil"]

#final_df = formatted_df[["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", "overall_sentiment"]]
#final_df.columns = ["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", "tgt_label"]

#    return final_df

#%% OVERALL SENTIMENT TEST RUN

#test_suffix = "Thumbs"
#final_train, final_dev, final_test = format_data(train, test_suffix), format_data(dev, test_suffix), format_data(test, test_suffix)



#%% TODO: for food --
# 1. format the labels, {0: negative, 1: positive}