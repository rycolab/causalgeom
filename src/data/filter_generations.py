#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv
import random

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import torch

#sys.path.append('../')
sys.path.append('./src/')

from paths import DATASETS, OUT

from utils.lm_loaders import GPT2_LIST, get_concept_name
#from utils.lm_loaders import GPT2_LIST
from data.embed_wordlists.embedder import get_token_list_outfile_paths
from data.process_generations_x import load_concept_token_lists

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")
   

#%% Filtering functions
def get_concept_hs(generations_folder, l0_tl, l1_tl, nsamples=None):
    files = os.listdir(generations_folder)
    if nsamples is not None:
        random.shuffle(files)
        files = files[:nsamples]
    l0_hs = []
    l1_hs = []
    for i, filepath in enumerate(tqdm(files)):
        with open(os.path.join(generations_folder, filepath), "rb") as f:
            data = pickle.load(f)

        for h, x in data:
            if x in l0_tl:
                l0_hs.append(h)
            elif x in l1_tl:
                l1_hs.append(h)
            else:
                continue
    return l0_hs, l1_hs

def get_concept_hs_w_factfoil(generations_folder, l0_tl, l1_tl, nsamples=None):
    files = os.listdir(generations_folder)
    if nsamples is not None:
        random.shuffle(files)
        files = files[:nsamples]
    l0_hs = []
    l1_hs = []
    for i, filepath in enumerate(tqdm(files)):
        with open(os.path.join(generations_folder, filepath), "rb") as f:
            data = pickle.load(f)

        for h, x in data:
            if x in l0_tl:
                x_index = np.nonzero(l0_tl==x)[0][0]
                foil = l1_tl[x_index]
                l0_hs.append((h.numpy(), x, foil))
            elif x in l1_tl:
                x_index = np.nonzero(l1_tl==x)[0][0]
                foil = l0_tl[x_index]
                l1_hs.append((h.numpy(), x, foil))
            else:
                continue
    return l0_hs, l1_hs

#%% Loader functions
def load_filtered_hs(model_name, I_P="no_I_P", nsamples=None):
    filtered_hs_dir = os.path.join(OUT, 
        f"filtered_generations/{model_name}/{I_P}")
    with open(os.path.join(filtered_hs_dir, "l0_hs.pkl"), "rb") as f:
        l0_hs = np.vstack(pickle.load(f))
    with open(os.path.join(filtered_hs_dir, "l1_hs.pkl"), "rb") as f:
        l1_hs = np.vstack(pickle.load(f))
    if nsamples is not None:
        np.random.shuffle(l0_hs)
        np.random.shuffle(l1_hs)
        ratio = l1_hs.shape[0]/l0_hs.shape[0]
        l0_hs = l0_hs[:nsamples,:]
        l1_hs = l1_hs[:int((nsamples*ratio)),:]
    return l0_hs, l1_hs

def load_filtered_hs_wff(model_name, I_P="no_I_P", nsamples=None):
    filtered_hs_dir = os.path.join(OUT, 
        f"filtered_generations/{model_name}/{I_P}")
    with open(os.path.join(filtered_hs_dir, "l0_hs_w_factfoil.pkl"), "rb") as f:
        l0_hs_wff = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "l1_hs_w_factfoil.pkl"), "rb") as f:
        l1_hs_wff = pickle.load(f)
    if nsamples is not None:
        random.shuffle(l0_hs_wff)
        random.shuffle(l1_hs_wff)
        ratio = len(l1_hs_wff)/len(l0_hs_wff)
        l0_hs_wff = l0_hs_wff[:nsamples]
        l1_hs_wff = l1_hs_wff[:int((nsamples*ratio))]
    return l0_hs_wff, l1_hs_wff

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
    args = get_args()
    logging.info(args)
    
    #nsamples=None
    #model_name = args.model
    #if args.useP:
    #    I_P = "I_P"
    #else:
    #    I_P = "no_I_P"
    model_name = "gpt2-large"
    I_P = "no_I_P"
    nfiles = 100

    generations_folder = os.path.join(OUT, f"generated_text/{model_name}/{I_P}")
    outdir = os.path.join(OUT, f"filtered_generations/{model_name}/{I_P}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created output directory {outdir}")
    
    concept_name = get_concept_name(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept_name, model_name)

    logging.info(f"Filtering generations for model {model_name} with {I_P}")

    #l0_hs, l1_hs = get_concept_hs(
    #    generations_folder, l0_tl, l1_tl, nsamples=nsamples
    #)
    
    #l0_outfile = os.path.join(outdir, "l0_hs.pkl")
    #l1_outfile = os.path.join(outdir, "l1_hs.pkl")
    #with open(l0_outfile, "wb") as f:
    #    pickle.dump(l0_hs, f, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(l1_outfile, "wb") as f:
    #    pickle.dump(l1_hs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    l0_hs_wff, l1_hs_wff = get_concept_hs_w_factfoil(
        generations_folder, l0_tl, l1_tl, nsamples=nfiles
    )
    
    l0_wff_outfile = os.path.join(outdir, "l0_hs_w_factfoil.pkl")
    l1_wff_outfile = os.path.join(outdir, "l1_hs_w_factfoil.pkl")
    with open(l0_wff_outfile, "wb") as f:
        pickle.dump(l0_hs_wff, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(l1_wff_outfile, "wb") as f:
        pickle.dump(l1_hs_wff, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    logging.info(f"Exported concept hs to {outdir}")
