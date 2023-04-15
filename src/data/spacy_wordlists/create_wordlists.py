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

sys.path.append('../../')

from paths import HF_CACHE, OUT


#%%
language = "fr"
OUT_DIR = os.path.join(OUT, f"wordlist")
OUT_FILE = os.path.join(OUT_DIR, f"{language}_new.pkl")

with open(OUT_FILE, 'rb') as f:      
    data = pickle.load(f)

# %%
VBZ = {}
VBP = {}
for k, v in data.items():
    for kp, vp in v["token_tag"].items():
        if kp.strip()=="VBZ" and (vp / v["count"]) > .5:
            VBZ[k] = vp
        elif kp.strip()=="VBP" and (vp / v["count"]) > .5:
            VBP[k] = vp
        else:
            continue



# %%
