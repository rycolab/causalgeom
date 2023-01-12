#%%
import os
import warnings
import logging
import coloredlogs
import argparse
from datetime import datetime
import shutil

import numpy as np
from tqdm import tqdm

from transformers import BertTokenizerFast, BertForMaskedLM
import torch

from paths import THESIS_DATA, HF_CACHE
from data.hidden_states import collect_hidden_states
from data.h_matrix import create_h_matrix
from data.compute_svd import get_hmat_sample, export_svd

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
else: 
    torch.device("cpu")
    logging.warning("No GPU found")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Collect hidden states')
    argparser.add_argument(
        "--seed",
        type=int,
        help="MultiBERTs seed for tokenizer and model"
    )
    argparser.add_argument(
        "--chkpt",
        type=str,
        choices=["0k", "20k", "40k", "60k", "80k", "100k", "120k", "140k", 
            "160k", "180k", "200k", "300k", "400k", "500k", "600k", "700k", 
            "800k", "900k", "1000k", "1100k", "1200k", "1300k", "1400k", 
            "1500k", "1600k", "1700k", "1800k", "1900k", "2000k", "none"],
        help="MultiBERTs checkpoint for tokenizer and model"
    )
    argparser.add_argument(
        "--h_nsamples",
        type=int,
        help="Number of hidden states to compute"
    )
    argparser.add_argument(
        "--svd_nsamples",
        type=int,
        help="Number of hidden states to use for SVD computation",
        default=100000
    )
    return argparser.parse_args()


#%%
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

args = get_args()
logging.info(args)

SEED = args.seed
CHKPT = args.chkpt
NSAMPLES = args.h_nsamples
SVD_NSAMPLES = args.svd_nsamples
HS_BATCH_SIZE = 8

if SVD_NSAMPLES > NSAMPLES:
    SVD_NSAMPLES = NSAMPLES

#%%
if CHKPT == "none":
    MODEL_NAME = f"google/multiberts-seed_{SEED}"
else:
    MODEL_NAME = f"google/multiberts-seed_{SEED}-step_{CHKPT}"

TOKENIZER = BertTokenizerFast.from_pretrained(
    MODEL_NAME, model_max_length=512
)
MODEL = BertForMaskedLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=HF_CACHE, 
    is_decoder=False
)

#%% Helpers
def create_run_dir(run_dir):
    assert not os.path.exists(run_dir), "Run output dir already exists"
    os.makedirs(run_dir)
    logging.info(f"Created directory: {run_dir}")

#%%############################################################################
# COLLECT HIDDEN STATES
###############################################################################
RUN_HS_DIR = os.path.join(
    THESIS_DATA, f"hidden_states/multiberts/{SEED}/{CHKPT}/{TIMESTAMP}"
)
create_run_dir(RUN_HS_DIR)

RUN_HMAT_DIR = os.path.join(
    THESIS_DATA, f"h_matrices/multiberts/{SEED}/{CHKPT}/{TIMESTAMP}"
)
create_run_dir(RUN_HMAT_DIR)

#%% WIKI VERSION
for dataset in ["wikipedia", "bookcorpus"]:
    RUN_HS_DIR_DS = os.path.join(RUN_HS_DIR, dataset)
    os.mkdir(RUN_HS_DIR_DS)

    logging.info(
        f"{MODEL_NAME} - exporting {str(NSAMPLES)} hidden states from "
        f"{dataset} into {RUN_HS_DIR_DS}."
    )

    collect_hidden_states(
        device,
        MODEL,
        TOKENIZER,
        dataset,
        NSAMPLES,
        RUN_HS_DIR_DS,
        HS_BATCH_SIZE
    )

    RUN_HMAT_DS_FILE = os.path.join(
        RUN_HMAT_DIR, f"h_{dataset}.npy"
    )

    create_h_matrix(RUN_HS_DIR_DS, RUN_HMAT_DS_FILE, delete_batch_dir=True)

# Delete run hidden state dir
shutil.rmtree(RUN_HS_DIR)
logging.info(f"Deleted run batch dir: {RUN_HS_DIR}")

#%% Compute SVD
SVD_DIR = os.path.join(
    THESIS_DATA, f"svd/multiberts/{SEED}/{CHKPT}"
)
if not os.path.exists(SVD_DIR):
    os.makedirs(SVD_DIR)

h_wiki_path = os.path.join(
    RUN_HMAT_DIR, f"h_wikipedia.npy"
)
h_book_path = os.path.join(
    RUN_HMAT_DIR, f"h_bookcorpus.npy"
)

h_wiki = np.load(h_wiki_path)
h_book = np.load(h_book_path)
h = np.concatenate([h_wiki, h_book], axis=0)

h = get_hmat_sample(h, SVD_NSAMPLES)

RUN_SVD_PATH = os.path.join(
    SVD_DIR, 
    f"svd_{TIMESTAMP}_nsamples{SVD_NSAMPLES}.pickle"
)
export_svd(h, RUN_SVD_PATH)

#%%############################################################################
# COLLECT HIDDEN STATES
###############################################################################

