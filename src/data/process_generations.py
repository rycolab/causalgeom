#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import torch

#sys.path.append('../../')
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import GPT2_LIST

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
def identify_matches(all_hs, new_hs):
    all_index = 0
    match_index = torch.ones(new_hs.shape[0]) * -1
    for i, new_h in enumerate(new_hs):
        stop = False
        while not stop:
            close = torch.isclose(all_hs[all_index,:], new_h, 1E-5).all().item()
            if close:
                match_index[i] = all_index
                stop = True
            elif ((new_h[0] > all_hs[all_index,0]).item() and 
                all_index < (all_hs.shape[0] - 1)):
                all_index += 1
            else:
                stop = True
    return match_index

def compute_updates(all_counts_shape, new_hs, new_counts, match_indices):
    count_update = torch.zeros(all_counts_shape)
    new_h_list = []
    new_count_list = []
    for index, new_h, count in zip(match_indices, new_hs, new_counts):
        index_ = int(index.item())
        count_ = int(count.item())
        if index_ != -1:
            count_update[index_] = count_
        else:
            new_h_list.append(new_h)
            new_count_list.append(count_)
    return count_update, torch.vstack(new_h_list), torch.tensor(new_count_list)

def merge_unique_counts(agg_hs, agg_counts, update_hs, update_counts):
    matched_indices = identify_matches(agg_hs, update_hs)
    count_update, new_hs, new_counts = compute_updates(
        agg_counts.shape, update_hs, update_counts, matched_indices
    )

    all_hs = torch.vstack((agg_hs, new_hs))
    all_counts = torch.hstack((agg_counts + count_update, new_counts))
    all_merged = torch.hstack((all_hs, all_counts.unsqueeze(1)))

    all_sorted = torch.unique(all_merged, dim=0)
    hs_sorted, counts_sorted = all_sorted[:,:-1], all_sorted[:,-1]
    return hs_sorted, counts_sorted

#%%
def create_temp_files(generationsdir, tempdir):
    files = os.listdir(generationsdir)
    all_hs, all_counts = None, None
    tempcount = 0
    tempbatches = 10
    for i, filepath in enumerate(tqdm(files)):
        with open(os.path.join(generationsdir, filepath), "rb") as f:
            data = pickle.load(f)

        hs = [x[0] for x in data]
        hs = torch.vstack(hs)
        unique_hs, counts = torch.unique(hs, return_counts=True, dim=0)
        if all_hs is None and all_counts is None:
            all_hs, all_counts = unique_hs, counts
        elif (i+1)%tempbatches == 0 or i == len(files)-1:
            all_hs, all_counts = merge_unique_counts(all_hs, all_counts, unique_hs, counts)
            tempfile = os.path.join(
                tempdir, 
                f"temp{tempcount}.pkl"
            )
            with open(tempfile, 'wb') as f:
                pickle.dump((all_hs, all_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Exported tempfile {tempcount}")
            all_hs, all_counts = None, None
            tempcount+=1
        else:
            all_hs, all_counts = merge_unique_counts(all_hs, all_counts, unique_hs, counts)

def aggregate_temp_files(tempdir, outfile):
    tempfiles = os.listdir(tempdir)
    all_hs, all_counts = None, None
    for i, filepath in enumerate(tqdm(tempfiles)):
        with open(os.path.join(tempdir, filepath), "rb") as f:
            file_hs, file_counts = pickle.load(f)

        if all_hs is None and all_counts is None:
            all_hs, all_counts = file_hs, file_counts
        else:
            all_hs, all_counts = merge_unique_counts(
                all_hs, all_counts, file_hs, file_counts
            )
    with open(outfile, 'wb') as f:
        pickle.dump((all_hs, all_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Exported all hs to {outfile}")
    
#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Generate from model')
    argparser.add_argument(
        "-model",
        type=str,
        choices=GPT2_LIST,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-useP",
        action="store_true",
        default=False,
        help="Whether to load and apply a P for this set of generations",
    )
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.info(args)
    
    model_name = args.model
    if args.useP:
        I_P = "I_P"
    else:
        I_P = "no_I_P"
    #model_name = "gpt2-large"
    #I_P = "I_P"
    generations_folder = os.path.join(OUT, f"generated_text/{model_name}/{I_P}")
    outdir = os.path.join(OUT, f"p_h/{model_name}")
    tempdir = os.path.join(outdir, "tempdir")
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    outfile = os.path.join(outdir, f"unique_hs_counts_{model_name}_{I_P}.pkl")

    logging.info(f"Processing generations for model {model_name} with {I_P}")
    
    create_temp_files(generations_folder, tempdir)
    
    logging.info("Tempfiles created, merging")
    aggregate_temp_files(tempdir, outfile)
