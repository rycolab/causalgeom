#%%
import sys
sys.path.append('..')

import warnings
import logging
import os
import coloredlogs
import argparse

import numpy as np
import seaborn as sns
import pickle
import pandas as pd

from paths import THESIS_DATA
from data.compute_svd import get_hmat_sample, export_svd

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%% Helpers
def load_svd_data(filepath):
    with open(filepath, 'rb') as file:
        svd_dict = pickle.load(file)
    return svd_dict["u"], svd_dict["s"]

def get_condition_number(s):
    return s[0] / s[-1]


#%%
SVD_DIR = "/cluster/scratch/cguerner/thesis_data/svd/multiberts"

SEED = 0
CHKPT_LIST = [
    "0k", "20k", "40k", "60k"#, "80k", "100k", "120k", "140k", 
    #"160k", "180k", "200k", "400k", "600k", "800k", "1000k", 
    #"1200k", "1400k",  "1600k", "1800k", "2000k"
]

condition_numbers = []
eigenvector_projections = []
for CHKPT in CHKPT_LIST:
    logging.info(f"----- SEED {SEED} CHKPT {CHKPT} -----")
    hmat_dir = os.path.join(
        THESIS_DATA, f"h_matrices/multiberts/{SEED}/{CHKPT}"
    )
    if not os.path.exists(hmat_dir):
        logging.warning("Hmat dir not found")
        break
    timestamp_folders = os.listdir(hmat_dir)
    timestamp = timestamp_folders[0]

    timestamp_dir = os.path.join(hmat_dir, timestamp)

    h_matrices = os.listdir(timestamp_dir)

    #%%
    SVD_DIR = os.path.join(
        THESIS_DATA, f"svd/multiberts/{SEED}/{CHKPT}"
    )
    if not os.path.exists(SVD_DIR):
        os.makedirs(SVD_DIR)

    h_wiki_path = os.path.join(
        timestamp_dir, f"h_wikipedia.npy"
    )
    h_book_path = os.path.join(
        timestamp_dir, f"h_bookcorpus.npy"
    )
    if ((not os.path.isfile(h_wiki_path)) or 
        (not os.path.isfile(h_book_path))):
        logging.warning("One of the H matrices missing")
        continue

    h_wiki = np.load(h_wiki_path)
    h_book = np.load(h_book_path)
    if ((not (h_wiki.shape[0] >= 150000 and h_wiki.shape[1] == 768)) or 
        (not(h_book.shape[0] >= 150000 and h_book.shape[1] == 768))):
        logging.warning("H wiki shape not correct")
        continue

    h = np.concatenate([h_wiki, h_book], axis=0)

    SVD_NSAMPLES = 100000
    RUN_SVD_PATH = os.path.join(
        SVD_DIR, 
        f"svd_{timestamp}_nsamples{SVD_NSAMPLES}.pickle"
    )

    u, s = load_svd_data(RUN_SVD_PATH)

    cn = get_condition_number(s)
    condition_numbers.append((CHKPT, cn))

    res = []
    for i in [0,-1]:
        res.append((f"{CHKPT}_{i}", h @ u[:,i]))
    eigenvector_projections.append(res)

#%%
condition_numbers

#%% 
for res in eigenvector_projections:
    for dist in res:
        sns.displot(dist[1], kind="hist").set(title=f"Distplot for {dist[0]}")

# %%
