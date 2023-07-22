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
from data.embed_wordlists.embedder import get_token_list_outfile_paths, load_concept_token_lists

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
    other_hs = []
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
                other_hs.append((h.numpy(), x))
                #continue
    return l0_hs, l1_hs, other_hs

#%% Loader functions
def sample_filtered_hs(l0_hs, l1_hs, nsamples):
    random.shuffle(l0_hs)
    random.shuffle(l1_hs)
    ratio = len(l1_hs)/len(l0_hs)
    if ratio > 1:
        l0_hs = l0_hs[:nsamples]
        l1_hs = l1_hs[:int((nsamples*ratio))]
    else:
        ratio = len(l0_hs) / len(l1_hs)
        l0_hs = l0_hs[:int((nsamples*ratio))]
        l1_hs = l1_hs[:nsamples]
    return l0_hs, l1_hs 

""" 
def load_filtered_hs(model_name, I_P="no_I_P", nsamples=None):
    filtered_hs_dir = os.path.join(OUT, 
        f"filtered_generations/{model_name}/{I_P}")
    with open(os.path.join(filtered_hs_dir, "l0_hs.pkl"), "rb") as f:
        l0_hs = np.vstack(pickle.load(f))
    with open(os.path.join(filtered_hs_dir, "l1_hs.pkl"), "rb") as f:
        l1_hs = np.vstack(pickle.load(f))
    if nsamples is not None:
        l0_hs, l1_hs = sample_filtered_hs(l0_hs, l1_hs, nsamples)
    return l0_hs, l1_hs
"""

def load_filtered_hs_wff(model_name, load_other=False, nucleus=False, nsamples=None, I_P="no_I_P"):
    if nucleus:
        filtered_hs_dir = os.path.join(OUT, 
            f"filtered_generations_nucleus/{model_name}/{I_P}")
    else:
        filtered_hs_dir = os.path.join(OUT, 
            f"filtered_generations/{model_name}/{I_P}"
        )
    if load_other:
        filtered_hs_dir = os.path.join(os.path.join(
            filtered_hs_dir, '..'), "no_I_P_w_other")
        #filtered_hs_dir = os.path.join(filtered_hs_dir, f"_w_other")
    
    with open(os.path.join(filtered_hs_dir, "l0_hs_w_factfoil.pkl"), "rb") as f:
        l0_hs_wff = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "l1_hs_w_factfoil.pkl"), "rb") as f:
        l1_hs_wff = pickle.load(f)
    
    if load_other:
        with open(os.path.join(filtered_hs_dir, "other_hs.pkl"), "rb") as f:
            other_hs = pickle.load(f)
        return l0_hs_wff, l1_hs_wff, other_hs
    else:
        if nsamples is not None:
            l0_hs_wff, l1_hs_wff = sample_filtered_hs(
                l0_hs_wff, l1_hs_wff, nsamples
            )
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
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.info(args)
    
    
    #model_name = args.model
    #nucleus = args.nucleus
    #nfiles=None
    model_name = "gpt2-large"
    nucleus = False
    nfiles = 10

    if nucleus:
        generations_folder = os.path.join(OUT, f"generated_text_nucleus/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"filtered_generations_nucleus/{model_name}/no_I_P_w_other")
    else:
        generations_folder = os.path.join(OUT, f"generated_text/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"filtered_generations/{model_name}/no_I_P_w_other")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created output directory {outdir}")
    
    concept_name = get_concept_name(model_name)
    l0_tl, l1_tl = load_concept_token_lists(concept_name, model_name)

    logging.info(f"Filtering generations for model {model_name} with nucleus: {nucleus}")

    #l0_hs, l1_hs = get_concept_hs(
    #    generations_folder, l0_tl, l1_tl, nsamples=nsamples
    #)
    
    #l0_outfile = os.path.join(outdir, "l0_hs.pkl")
    #l1_outfile = os.path.join(outdir, "l1_hs.pkl")
    #with open(l0_outfile, "wb") as f:
    #    pickle.dump(l0_hs, f, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(l1_outfile, "wb") as f:
    #    pickle.dump(l1_hs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    l0_hs_wff, l1_hs_wff, other_hs = get_concept_hs_w_factfoil(
        generations_folder, l0_tl, l1_tl, nsamples=nfiles
    )
    
    l0_wff_outfile = os.path.join(outdir, "l0_hs_w_factfoil.pkl")
    l1_wff_outfile = os.path.join(outdir, "l1_hs_w_factfoil.pkl")
    other_outfile = os.path.join(outdir, "other_hs.pkl")
    with open(l0_wff_outfile, "wb") as f:
        pickle.dump(l0_hs_wff, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(l1_wff_outfile, "wb") as f:
        pickle.dump(l1_hs_wff, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(other_outfile, "wb") as f:
        pickle.dump(other_hs, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported concept hs to {outdir}")
