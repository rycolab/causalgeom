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

from transformers import BertTokenizerFast, BertForMaskedLM

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC

#sys.path.append('..')
sys.path.append('./src/')

from utils.cuda_loaders import get_device
from utils.lm_loaders import get_model, get_tokenizer, get_V
from data.dataset_loaders import load_dataset
from paths import OUT, HF_CACHE

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
DATASET_NAME = "linzen"
DATASET_PATH = get_dataset_path(DATASET_NAME)
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/hidden_states/{DATASET_NAME}/{MODEL_NAME}"
BATCH_SIZE = 64

assert not os.path.exists(OUTPUT_DIR), \
    f"Hidden state export dir exists: {OUTPUT_DIR}"

os.mkdir(OUTPUT_DIR)


#%%
device = get_device()

TOKENIZER = get_tokenizer(MODEL_NAME)
MODEL = get_model(MODEL_NAME)
V = get_V(MODEL_NAME, MODEL)
MODEL = MODEL.to(device)

MASK_TOKEN_ID = TOKENIZER.mask_token_id

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

#%%
def format_batch_masked(batch_data, tokenizer, V):
    """ after obtaining hidden states, goes sentence by sentence and fetches
    fact and foil embeddings. 
    - adds a 1 to the hidden state
    """
    data = []
    for fact, foil, hs, tgt_label in zip(batch_data["fact"], 
                                batch_data["foil"], 
                                batch_data["hidden_states"],
                                batch_data["tgt_label"]):
        full_hs = np.append(hs, 1)

        fact_tok_ids = TOKENIZER.encode(fact)[1:-1]
        foil_tok_ids = TOKENIZER.encode(foil)[1:-1]

        if len(fact_tok_ids) == 1 and len(foil_tok_ids) == 1:
            fact_embedding = V[fact_tok_ids,:].flatten()
            foil_embedding = V[foil_tok_ids,:].flatten()
        else:
            fact_embedding = None
            foil_embedding = None
        
        data.append(dict(
            fact = fact,
            foil = foil,
            tgt_label = tgt_label,
            input_ids_fact = fact_tok_ids,
            fact_embedding = fact_embedding,
            input_ids_foil = foil_tok_ids,
            foil_embedding = foil_embedding,
            hs = full_hs
        ))
    return data


def export_batch(output_dir, batch_num, batch_data):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
    
    with open(export_path, 'wb') as file:
        pickle.dump(batch_data, file, protocol=pickle.HIGHEST_PROTOCOL)

#%%
"""
batch = next(iter(dl))
tokenized_text = TOKENIZER(
    batch["masked"], return_tensors='pt', 
    padding="max_length", truncation=True
)
batch_mask_indices = torch.where(
    (tokenized_text["input_ids"] == MASK_TOKEN_ID)
)
input_ids = tokenized_text["input_ids"].to(device)
token_type_ids = tokenized_text["token_type_ids"].to(device)
attention_mask = tokenized_text["attention_mask"].to(device)

with torch.no_grad():
    output = MODEL(
        input_ids=input_ids, token_type_ids=token_type_ids, 
        attention_mask=attention_mask, output_hidden_states=True
    )
    mlm_hs = MODEL.cls.predictions.transform(
        output["hidden_states"][-1]
    )

batch["hidden_states"] = mlm_hs[batch_mask_indices].cpu().detach().numpy()
formattedbatch, tokenizer_drops = format_batch_data(batch)
"""

#%%
def collect_hs_masked(dl, model, tokenizer, mask_token_id, V, output_dir):
    for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

        # Compute MASK hidden state
        tokenized_text = tokenizer(
            batch["masked"], return_tensors='pt', 
            padding="max_length", truncation=True
        )
        batch_mask_indices = torch.where(
            (tokenized_text["input_ids"] == mask_token_id)
        )
        input_ids = tokenized_text["input_ids"].to(device)
        token_type_ids = tokenized_text["token_type_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids, token_type_ids=token_type_ids, 
                attention_mask=attention_mask, output_hidden_states=True
            )
            mlm_hs = model.cls.predictions.transform(
                output["hidden_states"][-1]
            )

        #TODO:sanity check this with prob dist
        batch["hidden_states"] = mlm_hs[batch_mask_indices].cpu().detach().numpy()

        # Format and export batch
        batch = format_batch_masked(batch, tokenizer, V)
        export_batch(output_dir, i, batch)

        torch.cuda.empty_cache()

    logging.info(f"Finished exporting data to {output_dir}.")
