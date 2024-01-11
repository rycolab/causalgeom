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
from scipy.stats import entropy

#sys.path.append('../')
sys.path.append('./src/')

from paths import DATASETS, OUT

from utils.lm_loaders import GPT2_LIST, get_concept_name
from data.embed_wordlists.embedder import load_concept_token_lists


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
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    argparser.add_argument(
        "-nsamples",
        type=int,
        help="Number of files to read"
    )
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.info(args)
    
    model_name = args.model
    nucleus = args.nucleus
    nsamples = args.nsamples
    #model_name = "gpt2-large"
    #nucleus = True
    #nsamples = 20

    if nucleus:
        generations_folder = os.path.join(OUT, f"generated_text_nucleus/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"p_x/{model_name}")    
        xs_outfile = os.path.join(outdir, f"x_counts_{model_name}_nucleus.pkl")
        cs_outfile = os.path.join(outdir, f"c_counts_{model_name}_nucleus.pkl")
    else:
        generations_folder = os.path.join(OUT, f"generated_text/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"p_x/{model_name}")
        xs_outfile = os.path.join(outdir, f"x_counts_{model_name}.pkl")
        cs_outfile = os.path.join(outdir, f"c_counts_{model_name}.pkl")
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created output directory {outdir}")
    
    logging.info(f"Processing generations for model {model_name}")

    xs = get_xs_count_dict(generations_folder, nsamples)
    #%%
    with open(xs_outfile, "wb") as f:
        pickle.dump(xs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logging.info(f"Exported token count dict to: {xs_outfile}")

    cs = get_cs_count_dict(model_name, xs)
    
    with open(cs_outfile, "wb") as f:
        pickle.dump(cs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logging.info(f"Exported concept count dict to: {cs_outfile}")
