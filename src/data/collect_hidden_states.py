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

from paths import OUT, HF_CACHE, LINZEN_PREPROCESSED

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
DATASET_NAME = "linzen"
DATASET_PATH = LINZEN_PREPROCESSED
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/hidden_states/{DATASET_NAME}/{MODEL_NAME}"
BATCH_SIZE = 64

#%%
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
else: 
    torch.device("cpu")
    logging.warning("No GPU found")

#%%
TOKENIZER = BertTokenizerFast.from_pretrained(
    MODEL_NAME, model_max_length=512
)
MODEL = BertForMaskedLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=HF_CACHE, 
    is_decoder=False
)

MASK_TOKEN_ID = TOKENIZER.mask_token_id
word_embeddings = MODEL.bert.embeddings.word_embeddings.weight
bias = MODEL.cls.predictions.decoder.bias
V = torch.cat(
    (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()

MODEL = MODEL.to(device)

#%%
data = []
with open(DATASET_PATH) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        #masked = line[2].replace("***mask***", "[MASK]")
        masked = line[2]
        #mask_index = masked.split().index("[MASK]")
        sample = dict(
            masked=masked,
            #mask_index=mask_index,
            verb=line[3],
            iverb=line[4],
            verb_pos=line[5]
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

#%%
def get_masked_hs(batch_mlm_hs, batch_masked_token_indices):
    masked_hs = []
    for k, masked_index in enumerate(batch_masked_token_indices):
        masked_hs.append(batch_mlm_hs[k,masked_index,:])
    return torch.stack(masked_hs,axis=0).cpu().detach().numpy() 

def format_batch_data(batch_data):
    """ after obtaining hidden states, goes sentence by sentence and fetches
    verb and foil embeddings. 
    - doesn't include samples for verbs that tokenize to two tokens. 
    - adds a 1 to the hidden state
    """
    data = []
    count_tokenizer_drops = 0
    for i, tup in enumerate(zip(batch_data["verb"], 
                                batch_data["iverb"], 
                                batch_data["hidden_states"],
                                batch_data["verb_pos"])):
        verb = tup[0]
        iverb = tup[1]
        hidden_state = np.append(tup[2], 1)
        verb_pos = tup[3]

        verb_index = TOKENIZER.encode(verb)[1:-1]
        iverb_index = TOKENIZER.encode(iverb)[1:-1]

        if len(verb_index) == 1 and len(iverb_index) == 1:
            verb_embedding = V[verb_index,:].flatten()
            iverb_embedding = V[iverb_index,:].flatten()
            data.append(
                (hidden_state, verb_embedding, iverb_embedding, verb_pos)
            )
        else:
            count_tokenizer_drops+=1
            continue

    return data, count_tokenizer_drops

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
total_tokenizer_drops = 0 # samples dropped cuz verb/iverb tokenizes to 2 tokens
for i, batch in enumerate(pbar:=tqdm(dl)):
    pbar.set_description(f"Generating hidden states")

    # Compute MASK hidden state
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

    #TODO:sanity check this with prob dist
    batch["hidden_states"] = mlm_hs[batch_mask_indices].cpu().detach().numpy()

    # Format and export batch
    batch, tokenizer_drops = format_batch_data(batch)
    total_tokenizer_drops += tokenizer_drops
    export_batch(OUTPUT_DIR, i, batch)

    torch.cuda.empty_cache()

logging.info(f"Finished exporting data to {OUTPUT_DIR}.")
logging.info(f"Dropped {total_tokenizer_drops} obs cuz of token issue, "
             f"{total_tokenizer_drops*100 / len(ds)} percent of obs")

