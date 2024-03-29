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

from utils.lm_loaders import GPT2_LIST, SUPPORTED_AR_MODELS
from data.embed_wordlists.embedder import get_token_list_outfile_paths, \
    load_concept_token_lists
from data.data_utils import sample_filtered_hs

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

def get_generated_files(generations_folder):
    gen_output_folders = os.listdir(generations_folder)
    files = []
    for folder in gen_output_folders:
        folder_path = os.path.join(generations_folder, folder)
        folder_files = os.listdir(folder_path)
        for filename in folder_files:
            if filename.endswith(".pkl"):
                files.append(os.path.join(folder_path, filename))
    return files

def get_concept_hs_w_factfoil_singletoken(generations_folder, l0_tl, l1_tl, nfiles=None):
    files = get_generated_files(generations_folder) 
    logging.info(f"Found {len(files)} generated text files in main directory {generations_folder}")   
    
    if nfiles is not None:
        random.shuffle(files)
        files = files[:nfiles]
    l0_hs = []
    l1_hs = []
    other_hs = []
    for i, filepath in enumerate(tqdm(files)):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        for h, x, all_tokens in data:
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

def get_concept_hs_w_factfoil_multitoken(generations_folder, l0_tl, l1_tl, 
    nfiles=None, perc_samples=.1):
    files = get_generated_files(generations_folder) 
    logging.info(f"Found {len(files)} generated text files in main directory {generations_folder}")   

    if nfiles is not None:
        random.shuffle(files)
        files = files[:nfiles]
    l0_hs = []
    l1_hs = []
    other_hs = []
    multitoken_matches = 0
    for i, filepath in enumerate(tqdm(files)):
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        except pickle.UnpicklingError:
            continue
        else:
            random.shuffle(data)
            for h, x, all_tokens in data[:int(len(data)*perc_samples)]:
                all_tokens = all_tokens.tolist()
                added_to_concept = False
                for i in range(len(l0_tl)):
                    l0_wordtok = l0_tl[i]
                    l1_wordtok = l1_tl[i]
                    if all_tokens[-len(l0_wordtok):] == l0_wordtok:
                        # (h, fact, foil, all_tokens)
                        l0_hs.append((h.numpy(), l0_wordtok, l1_wordtok, all_tokens)) 
                        added_to_concept=True
                        if len(l0_wordtok) > 1:
                            multitoken_matches+=1
                        break    
                    elif all_tokens[-len(l1_wordtok):] == l1_wordtok:
                        # (h, fact, foil, all_tokens)
                        l1_hs.append((h.numpy(), l1_wordtok, l0_wordtok, all_tokens)) 
                        added_to_concept=True
                        if len(l1_wordtok) > 1:
                            multitoken_matches+=1
                        break
                    else:
                        continue
                if not added_to_concept:
                    other_hs.append(h.numpy())
                #NOTE: if memory issues, need to subsample concept AND other hs
                #if len(other_hs) > other_hs_max_length:
                #    random.shuffle(other_hs)
                #    other_hs = other_hs[:other_hs_max_length]
    logging.info(f"Generated h's obtained -- l0: {len(l0_hs)}, l1: {len(l1_hs)}, other: {len(other_hs)}.")
    logging.info(f"Number of multitoken matches: {multitoken_matches}")
    return l0_hs, l1_hs, other_hs


#%% Loader functions
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
def load_generated_hs_wff(model_name, concept, nucleus):
    if nucleus:
        filtered_hs_dir = os.path.join(OUT, f"filtered_generations_nucleus/{model_name}/{concept}")
    else:
        filtered_hs_dir = os.path.join(OUT, f"filtered_generations/{model_name}/{concept}")
    
    with open(os.path.join(filtered_hs_dir, "l0_hs_w_factfoil.pkl"), "rb") as f:
        l0_hs_wff = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "l1_hs_w_factfoil.pkl"), "rb") as f:
        l1_hs_wff = pickle.load(f)
    with open(os.path.join(filtered_hs_dir, "other_hs.pkl"), "rb") as f:
        other_hs = pickle.load(f)
    return l0_hs_wff, l1_hs_wff, other_hs

# %%
def get_args():
    argparser = argparse.ArgumentParser(description='Generate from model')
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["number", "gender", "food", "ambiance", "service", "noise"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    argparser.add_argument(
        "-perc_samples",
        type=float,
        default=.1,
        help="Percentage of h's stored in a given output file to analyze.",
    )
    argparser.add_argument(
        "-nfiles",
        type=int,
        default=None,
        help="Number of output files to load (should not be used outside debugging).",
    )
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.info(args)
    
    
    model_name = args.model
    concept = args.concept
    nucleus = args.nucleus
    perc_samples = args.perc_samples
    #model_name = "llama2"
    #concept = "food"
    #nucleus = True
    #perc_samples = .1
    
    #testing options
    nfiles = args.nfiles #keep this to none unless debugging
    #single_token = False

    if nucleus:
        generations_folder = os.path.join(OUT, f"generated_text_nucleus/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"filtered_generations_nucleus/{model_name}/{concept}")
    else:
        generations_folder = os.path.join(OUT, f"generated_text/{model_name}/no_I_P")
        outdir = os.path.join(OUT, f"filtered_generations/{model_name}/{concept}")

    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Created output directory {outdir}")
    
    l0_tl, l1_tl = load_concept_token_lists(concept, model_name, single_token=False)
    
    logging.info(f"Filtering generations for model {model_name} with nucleus: {nucleus}")
    
    #NOTE: no support for single_token anymore here
    l0_hs_wff, l1_hs_wff, other_hs = get_concept_hs_w_factfoil_multitoken(
        generations_folder, l0_tl, l1_tl, nfiles=nfiles, perc_samples=perc_samples
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
