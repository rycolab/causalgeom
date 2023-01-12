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

#sys.path.append('./src/')

from paths import OUT, HF_CACHE, BERT_SYNTAX_LINZEN_GOLDBERG

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16

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
MODEL = MODEL.to(device)

#%%
LGD_DATASET = "/cluster/work/cotterell/cguerner/bert-syntax/lgd_dataset.tsv"
OUTPUT_DIR = "/cluster/work/cotterell/cguerner/usagebasedprobing/out/lgd_dataset"
data = []
with open(LGD_DATASET) as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        masked = line[2].replace("***mask***", "[MASK]")
        mask_index = masked.split().index("[MASK]")
        sample = dict(
            masked = masked,
            mask_index = mask_index,
            verb = line[3],
            iverb = line[4]
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

def export_batch(output_dir, batch_num, batch_data):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
    
    with open(export_path, 'wb') as file:
        pickle.dump(batch_data, file, protocol=pickle.HIGHEST_PROTOCOL)


#%%
for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

    tokenized_text = TOKENIZER(
        batch["masked"], return_tensors='pt', 
        padding="max_length", truncation=True
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
    masked_lm_hs = get_masked_hs(mlm_hs, batch["mask_index"])
    batch["hidden_states"] = masked_lm_hs
    export_batch(OUTPUT_DIR, i, batch)

    torch.cuda.empty_cache()

logging.info(f"Finished exporting data to {OUTPUT_DIR}")