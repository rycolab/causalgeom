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

#sys.path.append('../')
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import GPT2_LIST
from data.embed_wordlists.embedder import get_token_list_outfile_paths

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")
   

#%%
def get_xs_count_dict(generations_folder, nsamples=None):
    files = os.listdir(generations_folder)
    if nsamples is not None:
        files = files[:nsamples]
    xs = {}
    for i, filepath in enumerate(tqdm(files)):
        with open(os.path.join(generations_folder, filepath), "rb") as f:
            data = pickle.load(f)

        for _, x in data:
            xcount = xs.get(x, 0)
            xcount+=1
            xs[x] = xcount
    return xs

#%%
def load_concept_token_lists(concept, model_name):
    _, l0_tl_file, l1_tl_file = get_token_list_outfile_paths(
        concept, model_name)
    #other_tl = np.load(other_tl_file)
    l0_tl = np.load(l0_tl_file)
    l1_tl = np.load(l1_tl_file)
    return l0_tl, l1_tl

def update_count(cdict, key, newcount):
    count = cdict.get(key, 0)
    count += newcount
    cdict[key] = count
    return cdict

def create_concept_counts(xs, l0_tl, l1_tl):
    c = {}
    for k, v in xs.items():
        if k in l0_tl:
            c = update_count(c, "l0", v)
        elif k in l1_tl:
            c = update_count(c, "l1", v)
        else:
            c = update_count(c, "other", v)
    return c

def get_concept_name(model_name):
    if model_name in ["gpt2-large", "bert-base-uncased"]:
        return "number"
    elif model_name in ["gpt2-base-french", "camembert-base"]:
        return "gender"
    else:
        raise ValueError(f"No model to concept mapping")

def get_cs_count_dict(model_name, xs):
    concept_name = get_concept_name(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept_name, model_name)
    return create_concept_counts(xs, l0_tl, l1_tl)

# %%
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
    #args = get_args()
    #logging.info(args)
    
    #model_name = args.model
    #if args.useP:
    #    I_P = "I_P"
    #else:
    #    I_P = "no_I_P"
    model_name = "gpt2-large"
    I_P = "I_P"
    nsamples = 20

    generations_folder = os.path.join(OUT, f"generated_text/{model_name}/{I_P}")
    outdir = os.path.join(OUT, f"p_x/{model_name}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created output directory {outdir}")
    xs_outfile = os.path.join(outdir, f"x_counts_{model_name}_{I_P}.pkl")
    cs_outfile = os.path.join(outdir, f"c_counts_{model_name}_{I_P}.pkl")

    logging.info(f"Processing generations for model {model_name} with {I_P}")

    xs = get_xs_count_dict(generations_folder, nsamples)
    #%%
    with open(xs_outfile, "wb") as f:
        pickle.dump(xs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logging.info(f"Exported token count dict to: {xs_outfile}")

    cs = get_cs_count_dict(model_name, xs)
    
    with open(cs_outfile, "wb") as f:
        pickle.dump(cs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logging.info(f"Exported concept count dict to: {cs_outfile}")
