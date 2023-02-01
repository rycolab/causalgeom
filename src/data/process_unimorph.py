#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from transformers import GPT2TokenizerFast

#import torch
#from torch.utils.data import DataLoader, Dataset
#from abc import ABC

sys.path.append('..')
#sys.path.append('./src/')

from paths import OUT, UNIMORPH_ENG

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
def create_bin(vals, morph):
    return all(i in morph for i in vals)

#%%
data = []
with open(UNIMORPH_ENG) as f:
    tsv_file = csv.reader(f, delimiter="\t")
    for line in tsv_file:
        morph = line[2]
        morph_list = morph.split(";")
        data.append(dict(
            lemma = line[0],
            word = line[1],
            morph = morph,
            #sg = create_bin(["V", "PRS", "3", "SG"], morph),
            #pl = create_bin(["V", "NFIN", "IMP+SBJV"], morph)
            sg = ["V", "PRS", "3", "SG"] == morph_list,
            pl = ["V", "NFIN", "IMP+SBJV"] == morph_list
        ))

# %%
df = pd.DataFrame(data)
# %%
df_nodup = df.drop_duplicates()
# %%
vc = df_nodup[((df_nodup["sg"] == 1) | (df_nodup["pl"] == 1))]["lemma"].value_counts()
vc[vc == 2]