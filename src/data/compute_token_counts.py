#%%
#TODO: REFACTOR FOR USE BY OTHERS

import warnings
import logging
import os
import coloredlogs
import argparse


import numpy as np
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset
from transformers import BertTokenizerFast

from utils import compute_unique_counts, merge_token_ids, get_dataset

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='ComputeTokenCounts')
    argparser.add_argument(
        "--dataset", 
        type=str,
        choices=["wikipedia", "bookcorpus"],
        required=True,
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "--percent",
        type=float,
        help="Percent of the dataset to sample"
    )
    argparser.add_argument(
        "--seed",
        type=int,
        help="MultiBERTs seed for tokenizer"
    )
    return argparser.parse_args()

#%% config
args = get_args()
logging.info(args)

DATASET = args.dataset # "wikipedia", "bookcorpus"
PERCENT = args.percent
SEED = args.seed
TOKENIZER = BertTokenizerFast.from_pretrained(
    f'google/multiberts-seed_{SEED}'
)
EXPORT_DIR = "/cluster/scratch/cguerner/thesis_data/unigram_freqs/"

assert os.path.exists(EXPORT_DIR), "Export dir doesn't exist"

assert PERCENT > 0 and PERCENT <= 1, "PERCENT not supported"
percent_filename = str(PERCENT)
percent_filename = percent_filename.replace(".", "_")

OUTFILE_PATH = os.path.join(
    EXPORT_DIR, 
    f"{DATASET}_{percent_filename}_percent.npy"
)
logging.info(f"Computing unigram distribution for {DATASET} using "
             f"{str(PERCENT)} of the dataset.")

#%%
ds = get_dataset(DATASET)

#%% Computing unigram
dim = len(ds["train"])

sample = np.random.choice(dim, round(PERCENT*dim), replace=False)

sample_ids = np.zeros(0, dtype=np.int64)
token_ids = np.arange(TOKENIZER.vocab_size, dtype=np.int64)
counts = np.zeros(TOKENIZER.vocab_size, dtype=np.int64)
for i, ind in enumerate(pbar:=tqdm(sample)):
    pbar.set_description(f"Processing random samples from dataset")
    text = ds["train"][int(ind)]["text"]
    encoded_input = TOKENIZER(text, return_tensors='pt')
    ids = np.array(encoded_input["input_ids"][0])
    sample_ids = np.concatenate([sample_ids, ids])
    if i % 100 == 0 or i == (len(sample)-1):
        new_tokenids, new_counts = compute_unique_counts(sample_ids)
        token_ids, counts = merge_token_ids(token_ids, counts, new_tokenids, new_counts)
        sample_ids = np.zeros(0, dtype=np.int64)

np.save(OUTFILE_PATH, np.stack((token_ids, counts)))

logging.info(f"Exported data to {OUTFILE_PATH}.")
