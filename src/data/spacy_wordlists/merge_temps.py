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

#sys.path.append('../../')

from paths import HF_CACHE, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#######################
# Processing batch files #
##########################
language = "fr"
OUT_DIR = os.path.join(OUT, f"wordlist")
TEMP_DIR = os.path.join(OUT_DIR, f"temp_{language}")
OUT_FILE = os.path.join(OUT_DIR, f"{language}.pkl")
files = os.listdir(TEMP_DIR)
files.sort()

#if nbatches is not None:
#    files = files[:nbatches]
#%%
def increment_feature_count(feature_dict, token_feature_val):
    feature_count = feature_dict.get(token_feature_val, 0)
    feature_dict[token_feature_val] = feature_count + 1
    return feature_dict

tokens_dict = {}
for i, filename in enumerate(tqdm(files)):
#i = 0 
#filename = files[0]
    filepath = os.path.join(TEMP_DIR, filename)
    with open(filepath, 'rb') as f:      
        data = pickle.load(f)

    for token in data:
        tt, tpos, ttag = token["text"].lower(), token["pos"], token["tag"]
        td = tokens_dict.get(tt, dict(token_pos = {}, token_tag = {}, count = 0))
        td["count"] += 1
        td["token_pos"] = increment_feature_count(td["token_pos"], tpos)
        td["token_tag"] = increment_feature_count(td["token_tag"], ttag)
        tokens_dict[tt] = td

with open(OUT_FILE, 'wb') as f:
    pickle.dump(tokens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
