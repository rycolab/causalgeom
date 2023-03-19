#TODO:
# - debug the new device, get_tokenizer, etc. functions

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
import pickle

from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC

#sys.path.append('..')
sys.path.append('./src/')

from paths import OUT, HF_CACHE
from utils.cuda_loaders import get_device
from utils.lm_loaders import get_model, get_tokenizer, get_V
from data.dataset_loaders import load_dataset

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
DATASET_NAME = "linzen"
DATASET_PATH = get_dataset_path(DATASET_NAME)
MODEL_NAME = "gpt2"
OUTPUT_DIR = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/hidden_states/{DATASET_NAME}/{MODEL_NAME}"
BATCH_SIZE = 64

assert not os.path.exists(OUTPUT_DIR), \
    f"Hidden state export dir exists: {OUTPUT_DIR}"

os.mkdir(OUTPUT_DIR)

#%%
device = get_device()

TOKENIZER = get_tokenizer(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
PAD_TOKEN_ID = TOKENIZER.encode(TOKENIZER.pad_token)[0]

MODEL = get_model(MODEL_NAME)

V = get_V(MODEL_NAME, MODEL)

MODEL = MODEL.to(device)

#%%
#TODO: need to add this split thing for UD
data = load_dataset(DATASET_NAME, MODEL_NAME)

#%%
class CustomDataset(Dataset, ABC):
    def __init__(self, list_of_samples):
        self.data = list_of_samples
        self.n_instances = len(self.data)
        
    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.data[index]

#%%
ds = CustomDataset(data)
dl = DataLoader(dataset = ds, batch_size=BATCH_SIZE)

#%% Helpers
def export_batch(output_dir, batch_num, batch_data):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
    
    with open(export_path, 'wb') as file:
        pickle.dump(batch_data, file, protocol=pickle.HIGHEST_PROTOCOL)

def batch_tokenize_tgts(tgts):
    return TOKENIZER(tgts)["input_ids"]

def batch_tokenize_text(text):
    return TOKENIZER(
        text, return_tensors='pt', 
        padding="max_length", truncation=True
    )

def get_raw_sample_hs(pre_tgt_ids, attention_mask, tgt_ids):
    nopad_ti = pre_tgt_ids[attention_mask == 1]
    tgt_ti = torch.cat((nopad_ti, torch.LongTensor(tgt_ids)), 0)
    tgt_ti_dev = tgt_ti.to(device)
    with torch.no_grad():
        output = MODEL(
            input_ids=tgt_ti_dev, 
            #attention_mask=attention_mask, 
            labels=tgt_ti_dev,
            output_hidden_states=True
        )
    return tgt_ti.numpy(), output.hidden_states[-1].cpu().numpy()


def get_tgt_hs(raw_hs, tgt_tokens):
    #pre_verb_hs = raw_hs[:-(len(tgt_tokens) + 1)]
    #verb_hs = raw_hs[-(tgt_tokens.shape[0]+1):]
    tgt_hs = raw_hs[-(len(tgt_tokens)+1):]
    return tgt_hs

def get_batch_hs(batch):
    batch_hs = []
    tok_facts = batch_tokenize_tgts(batch["fact"])
    tok_foils = batch_tokenize_tgts(batch["foil"])
    tok_text = batch_tokenize_text(batch["pre_tgt_text"])

    for ti, am, tfa, tfo, fact, foil, tgt_label in zip(
        tok_text["input_ids"], 
        tok_text["attention_mask"], 
        tok_facts,
        tok_foils,
        batch["fact"], 
        batch["foil"], 
        batch["tgt_label"]):
        
        ######################
        # Get hidden states
        ######################
        # NOTE: for dynamic program will need to separately
        # loop through all tokenizations of verb and iverb
        fact_ti, fact_raw_hs = get_raw_sample_hs(ti, am, tfa)
        fact_hs = get_verb_hs(fact_raw_hs, tfa)

        foil_ti, foil_raw_hs = get_raw_sample_hs(ti, am, tfo)
        foil_hs = get_verb_hs(foil_raw_hs, tfo)

        ######################
        # Get verb embeddings
        ######################
        fact_embedding = V[tfa,:]
        foil_embedding = V[tfo,:]

        batch_hs.append(dict(
            fact = fact,
            foil = foil,
            tgt_label = tgt_label,
            input_ids_pre_tgt = ti,
            input_ids_fact = tfa, 
            fact_hs = fact_hs, 
            #verb_raw_hs = fact_raw_hs,
            fact_embedding = fact_embedding,
            input_ids_foil = tfo, 
            foil_hs = foil_hs,
            #iverb_raw_hs=iverb_raw_hs,
            foil_embedding = foil_embedding,
        ))
    return batch_hs


for i, batch in enumerate(pbar:=tqdm(dl)):
    pbar.set_description(f"Generating hidden states")

    batch_data = get_batch_hs(batch)
    export_batch(OUTPUT_DIR, i, batch_data)

    torch.cuda.empty_cache()

logging.info(f"Finished exporting data to {OUTPUT_DIR}.")
#logging.info(f"Dropped {total_tokenizer_drops} obs cuz of token issue, "
#             f"{total_tokenizer_drops*100 / len(ds)} percent of obs")



""" OLD CODE
def get_ar_hs(batch_mlm_hs, batch_masked_token_indices):
    masked_hs = []
    for k, masked_index in enumerate(batch_masked_token_indices):
        masked_hs.append(batch_mlm_hs[k,masked_index,:])
    return torch.stack(masked_hs,axis=0).cpu().detach().numpy() 
"""