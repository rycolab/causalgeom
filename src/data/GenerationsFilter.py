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
from data.spacy_wordlists.embedder import load_concept_token_lists
from data.generation_filter_utils import process_sample

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")
   

#%%
class GenerationsFilter:

    def __init__(
        self, 
        model_name: str, 
        concept: str, 
        nucleus: bool, 
        perc_samples: float,
        nfiles=None
    ):

        self.perc_samples=perc_samples
        self.nfiles = nfiles

        # Directory handling
        if nucleus:
            self.generations_folder = os.path.join(
                OUT, f"generated_text_nucleus/{model_name}/no_I_P"
            )
            self.outdir = os.path.join(
                DATASETS, f"filtered_generations/nucleus/{model_name}/{concept}"
            )
        else:
            self.generations_folder = os.path.join(
                OUT, f"generated_text/{model_name}/no_I_P"
            )
            self.outdir = os.path.join(
                DATASETS, f"filtered_generations/ancestral/{model_name}/{concept}"
            )

        os.makedirs(self.outdir, exist_ok=True)
        logging.info(f"Created output directory {self.outdir}")
        
        # Token list loading
        self.l0_tl, self.l1_tl, _ = load_concept_token_lists(
            concept, model_name, single_token=False
        )

    @staticmethod
    def get_generated_files(generations_folder, nfiles):
        gen_output_folders = os.listdir(generations_folder)
        files = []
        for folder in gen_output_folders:
            folder_path = os.path.join(generations_folder, folder)
            folder_files = os.listdir(folder_path)
            for filename in folder_files:
                if filename.endswith(".pkl"):
                    files.append(os.path.join(folder_path, filename))
        
        logging.info(
            f"Found {len(files)} generated text files"
            f" in main directory {generations_folder}"
        )   

        if nfiles is not None:
            random.shuffle(files)
            files = files[:nfiles]
            logging.info(f"Applied nfiles = {nfiles} subsampling")

        return files

    def process_file_data(self, file_data):
        l0_samples, l1_samples, other_samples = [], [], []

        idx = np.arange(len(file_data))
        np.random.shuffle(idx)
        nsamples = int(len(file_data)*self.perc_samples)

        for i in idx[:nsamples]:
            h, x, all_tokens = file_data[i]

            match_type, sample_data = process_sample(
                i, h, all_tokens.tolist(), file_data, 
                self.l0_tl, self.l1_tl
            )
            if match_type == "l0" and sample_data[0] is not None:
                l0_samples.append(sample_data)
            elif match_type == "l1" and sample_data[0] is not None:
                l1_samples.append(sample_data)
            elif match_type == "other":
                other_samples.append(
                    (sample_data[0], sample_data[3]) #h, cxt_tok
                ) 
            else:
                raise ValueError(f"Incorrect match type {match_type}")
        return l0_samples, l1_samples, other_samples

    def filter_generations(self):
        files = self.get_generated_files(
            self.generations_folder, self.nfiles
        ) 

        l0_samples = []
        l1_samples = []
        other_samples = []
        multitoken_matches = 0
        for filepath in tqdm(files):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                continue
            else:
                file_samples = self.process_file_data(data)
                l0_samples = l0_samples + file_samples[0]
                l1_samples = l1_samples + file_samples[1]
                other_samples = other_samples + file_samples[2]
                
        logging.info(f"Generated samples obtained: "
                        f"- l0: {len(l0_samples)}\n" 
                        f"- l1: {len(l1_samples)}\n"
                        f"- other: {len(other_samples)}")
        return l0_samples, l1_samples, other_samples


#%% Loader functions
def load_filtered_generations(model_name, concept, nucleus):
    if nucleus:
        filtered_hs_dir = os.path.join(
            DATASETS, f"filtered_generations/nucleus/{model_name}/{concept}"
        )
    else:
        filtered_hs_dir = os.path.join(
            DATASETS, f"filtered_generations/ancestral/{model_name}/{concept}"
        )
    
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
        default=.05,
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
    #nfiles = None
    #single_token = False

    genfilter = GenerationsFilter(
        model_name, concept, nucleus, perc_samples, nfiles
    )
    l0_samples, l1_samples, other_samples = genfilter.filter_generations()
    
    l0_wff_outfile = os.path.join(genfilter.outdir, "l0_hs_w_factfoil.pkl")
    l1_wff_outfile = os.path.join(genfilter.outdir, "l1_hs_w_factfoil.pkl")
    other_outfile = os.path.join(genfilter.outdir, "other_hs.pkl")
    with open(l0_wff_outfile, "wb") as f:
        pickle.dump(l0_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(l1_wff_outfile, "wb") as f:
        pickle.dump(l1_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(other_outfile, "wb") as f:
        pickle.dump(other_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported concept hs to {genfilter.outdir}")
