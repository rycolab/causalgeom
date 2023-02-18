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

from paths import OUT, HF_CACHE, LINZEN_PREPROCESSED
from utils.cuda_loaders import get_device
from utils.lm_loaders import get_model, get_tokenizer, get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
DATASET_NAME = "linzen"
DATASET_PATH = LINZEN_PREPROCESSED
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
data = []
with open(DATASET_PATH) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        unmasked_text = line[1]
        verb = line[3]
        iverb = line[4]
        verb_pos = line[5]
        vindex = int(line[6])
        if vindex > 0:
            verb = " " + verb
            iverb = " " + iverb
        pre_verb_text = " ".join(unmasked_text.split(" ")[:vindex])
        verb_text = " ".join(unmasked_text.split(" ")[:(vindex+1)])
        iverb_text = " ".join(unmasked_text.split(" ")[:vindex] + [iverb])
        sample = dict(
            pre_verb_text=pre_verb_text,
            verb_text=verb_text,
            iverb_text=iverb_text,
            verb=verb,
            iverb=iverb,
            verb_pos=verb_pos
        )
        data.append(sample)

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

"""
def batch_tokenize_verbs(verbs):
    list_output = TOKENIZER(verbs)["input_ids"]
    tensor_list = [torch.LongTensor(x) for x in list_output]
    return torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_TOKEN_ID
    )
"""
def batch_tokenize_verbs(verbs):
    return TOKENIZER(verbs)["input_ids"]

def batch_tokenize_text(text):
    return TOKENIZER(
        text, return_tensors='pt', 
        padding="max_length", truncation=True
    )

"""
def get_raw_sample_hs(pre_verb_ids, attention_mask, verb_ids):
    nopad_ti = pre_verb_ids[attention_mask == 1]
    #nopad_verb_ids = verb_ids[verb_ids != PAD_TOKEN_ID]
    verb_ti = torch.cat((nopad_ti, verb_ids), 0)
    with torch.no_grad():
        output = MODEL(
            input_ids=verb_ti, 
            #attention_mask=attention_mask, 
            labels=verb_ti,
            output_hidden_states=True
        )
    return verb_ti, output.hidden_states[-1]
"""
def get_raw_sample_hs(pre_verb_ids, attention_mask, verb_ids):
    nopad_ti = pre_verb_ids[attention_mask == 1]
    verb_ti = torch.cat((nopad_ti, torch.LongTensor(verb_ids)), 0)
    verb_ti_dev = verb_ti.to(device)
    with torch.no_grad():
        output = MODEL(
            input_ids=verb_ti_dev, 
            #attention_mask=attention_mask, 
            labels=verb_ti_dev,
            output_hidden_states=True
        )
    return verb_ti.numpy(), output.hidden_states[-1].cpu().numpy()


def get_verb_hs(raw_hs, verb_tokens):
    #pre_verb_hs = raw_hs[:-(len(verb_tokens) + 1)]
    #verb_hs = raw_hs[-(verb_tokens.shape[0]+1):]
    verb_hs = raw_hs[-(len(verb_tokens)+1):]
    return verb_hs

def get_batch_hs(batch):
    batch_hs = []
    tok_verbs = batch_tokenize_verbs(batch["verb"])
    tok_iverbs = batch_tokenize_verbs(batch["iverb"])
    tok_text = batch_tokenize_text(batch["pre_verb_text"])

    for ti, am, tv, tiv, verb, iverb, verb_pos in zip(
        tok_text["input_ids"], 
        tok_text["attention_mask"], 
        tok_verbs,
        tok_iverbs,
        batch["verb"], 
        batch["iverb"], 
        batch["verb_pos"]):
        
        ######################
        # Get hidden states
        ######################
        # NOTE: for dynamic program will need to separately
        # loop through all tokenizations of verb and iverb
        verb_ti, verb_raw_hs = get_raw_sample_hs(ti, am, tv)
        verb_hs = get_verb_hs(verb_raw_hs, tv)

        iverb_ti, iverb_raw_hs = get_raw_sample_hs(ti, am, tiv)
        iverb_hs = get_verb_hs(iverb_raw_hs, tiv)

        ######################
        # Get verb embeddings
        ######################
        verb_embedding = V[tv,:]
        iverb_embedding = V[tiv,:]

        batch_hs.append(dict(
            verb = verb,
            iverb = iverb,
            verb_pos = verb_pos,
            input_ids_pre_verb = ti,
            input_ids_verb = tv, 
            verb_hs = verb_hs, 
            verb_embedding = verb_embedding,
            input_ids_iverb = tiv, 
            iverb_hs = iverb_hs,
            iverb_embedding = iverb_embedding)
        )
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