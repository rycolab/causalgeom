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

from transformers.activations import gelu

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC

sys.path.append('..')
#sys.path.append('./src/')

from paths import OUT, HF_CACHE, FR_DATASETS
from utils.cuda_loaders import get_device
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.dataset_loaders import load_wikipedia

import ipdb

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# General Helpers  #
####################
def export_batch(output_dir, batch_num, batch_data):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
    
    with open(export_path, 'wb') as file:
        pickle.dump(batch_data, file, protocol=pickle.HIGHEST_PROTOCOL)

#%%#######################
# Random Masking Helpers #
##########################
def mask_random_token(input_ids, attention_mask, mask_token_id):
    """ Takes input_ids and attention_mask, replaces one non-padded
    token with [MASK] and outputs the mask token index.
    """
    labels = input_ids.clone()
    
    nopad_input_ids = torch.index_select(
        input_ids, 0,index=(attention_mask==1).nonzero().squeeze()
    )
    mask_token_index = np.random.randint(nopad_input_ids.shape[0])
    input_ids[mask_token_index] = mask_token_id

    #labels = torch.where(input_ids == mask_token_id, labels, -100)
    return input_ids, mask_token_index#, labels

def mask_batch(batch_input_ids, batch_attention_mask, mask_token_id):
    n_samples = batch_input_ids.shape[0]

    batch_masked_ids = []
    batch_masked_token_indices = []
    #for j in range(n_samples):
    for sample, mask in zip(batch_input_ids, batch_attention_mask):
        #sample = batch_input_ids[j,:]
        #mask = batch_attention_mask[j,:]
        masked_sample, mask_token_index = mask_random_token(
            sample, mask, mask_token_id
        )
        batch_masked_ids.append(masked_sample)
        batch_masked_token_indices.append(mask_token_index)

    batch_masked_input_ids = torch.stack(batch_masked_ids, axis=0)
    return batch_masked_input_ids, batch_masked_token_indices

#def batch_tokenize_tgts(tgts, tokenizer):
#    return tokenizer(tgts)["input_ids"]


#def get_raw_sample_hs(pre_tgt_ids, attention_mask, tgt_ids, model):
#    nopad_ti = pre_tgt_ids[attention_mask == 1]
#    tgt_ti = torch.cat((nopad_ti, torch.LongTensor(tgt_ids)), 0)
#    tgt_ti_dev = tgt_ti.to(device)
#   with torch.no_grad():
#        output = model(
##            input_ids=tgt_ti_dev, 
#             #attention_mask=attention_mask, 
#            labels=tgt_ti_dev,
#            output_hidden_states=True
#        )
#    return tgt_ti.numpy(), output.hidden_states[-1].cpu().numpy()

def get_tgt_hs(raw_hs, tgt_tokens):
    #pre_verb_hs = raw_hs[:-(len(tgt_tokens) + 1)]
    #verb_hs = raw_hs[-(tgt_tokens.shape[0]+1):]
    #ipdb.set_trace()
    tgt_hs = raw_hs[-(len(tgt_tokens)+1):]
    return tgt_hs


def collect_hs_ar(dl, model, tokenizer, output_dir):

    for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

        tokenized_text = tokenizer(
            batch["masked"], return_tensors='pt', 
            padding="max_length", truncation=True
        )

        input_ids = tokenized_text["input_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=input_ids,
                output_hidden_states=True
            )
            hs = output.hidden_states[-1].cpu().numpy()

        batch_data = get_batch_hs_ar(batch, model, tokenizer, V)
        export_batch(output_dir, i, batch_data)

        torch.cuda.empty_cache()
    logging.info(f"Finished exporting data to {output_dir}.")

#%%#################
# Masked Helpers   #
####################
def collect_hs_masked(dl, model_name, model, tokenizer, mask_token_id, output_dir):
    for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

        # Compute MASK hidden state
        tokenized_text = tokenizer(
            batch["text"], return_tensors='pt', 
            padding="max_length", truncation=True
        )

        masked_input_ids, masked_token_indices = mask_batch(
            tokenized_text["input_ids"], 
            tokenized_text["attention_mask"],
            mask_token_id
        )

        #ipdb.set_trace()
        batch_mask_indices = torch.where(
            (masked_input_ids == mask_token_id)
        )
        input_ids = masked_input_ids.to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)
        if model_name=="bert-base-uncased":
            token_type_ids = tokenized_text["token_type_ids"].to(device)

        with torch.no_grad():
            if model_name=="bert-base-uncased":
                output = model(
                    input_ids=input_ids, token_type_ids=token_type_ids, 
                    attention_mask=attention_mask, output_hidden_states=True
                )
                mlm_hs = model.cls.predictions.transform(
                    output["hidden_states"][-1]
                )
            elif model_name=="camembert-base":
                output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                last_layer_hs = output["hidden_states"][-1]
                mlm_hs = model.lm_head.layer_norm(gelu(model.lm_head.dense(last_layer_hs)))

        batch["hidden_states"] = mlm_hs[batch_mask_indices].cpu().detach().numpy()

        # Format and export batch
        batch = format_batch_masked(batch, tokenizer, V)
        export_batch(output_dir, i, batch)

        torch.cuda.empty_cache()

    logging.info(f"Finished exporting data to {output_dir}.")


#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "-language", 
        type=str,
        choices=["fr", "en"],
        help="Language to collect hidden states for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=BERT_LIST + GPT2_LIST,
        help="Model for computing hidden states"
    )
    return argparser.parse_args()


#%%
#if __name__=="__main__":
#    args = get_args()
#    logging.info(args)

#language = args.language
#model_name = args.model
language = "en"
model_name = "gpt2"
#split = "dev"
batch_size = 10

# Load model, tokenizer
device = get_device()

tokenizer = get_tokenizer(model_name)
#pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
model = get_model(model_name)
#V = get_V(model_name, model)

model = model.to(device)

# load data
#data = load_dataset(dataset_name, model_name, split)
data = load_wikipedia(language)
#data.shuffle()
dl = DataLoader(dataset = data, batch_size=batch_size, shuffle=True)

testbatch = next(iter(dl))

tokenized_text = tokenizer(
    testbatch["text"], return_tensors='pt', 
    padding="max_length", truncation=True
)

input_ids = tokenized_text["input_ids"].to(device)
attention_mask = tokenized_text["attention_mask"].to(device)

with torch.no_grad():
    output = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=input_ids,
        output_hidden_states=True
    )
    hs = output.hidden_states[-1].cpu().numpy()










mask_token_id = tokenizer.mask_token_id
tokenized_text = tokenizer(
    testbatch["text"], return_tensors='pt', 
    padding="max_length", truncation=True
)

masked_input_ids, masked_token_indices = mask_batch(
    tokenized_text["input_ids"], 
    tokenized_text["attention_mask"],
    mask_token_id    
)

#ipdb.set_trace()
batch_mask_indices = torch.where(
    (masked_input_ids == mask_token_id)
)
input_ids = masked_input_ids.to(device)
attention_mask = tokenized_text["attention_mask"].to(device)
if model_name=="bert-base-uncased":
    token_type_ids = tokenized_text["token_type_ids"].to(device)

with torch.no_grad():
    if model_name=="bert-base-uncased":
        output = model(
            input_ids=input_ids, token_type_ids=token_type_ids, 
            attention_mask=attention_mask, output_hidden_states=True
        )
        mlm_hs = model.cls.predictions.transform(
            output["hidden_states"][-1]
        )
    elif model_name=="camembert-base":
        output = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        last_layer_hs = output["hidden_states"][-1]
        mlm_hs = model.lm_head.layer_norm(gelu(model.lm_head.dense(last_layer_hs)))

testbatch["hidden_states"] = mlm_hs[batch_mask_indices].cpu().detach().numpy()





# Output dir
output_dir = os.path.join(OUT, f"hidden_states/{dataset_name}/{model_name}")

assert not os.path.exists(output_dir), \
    f"Hidden state export dir exists: {output_dir}"

os.makedirs(output_dir)

# Collect HS
if model_name in GPT2_LIST:
    logging.info(f"Collecting hs for model {model_name} in AR mode.")
    collect_hs_ar(dl, model, tokenizer, V, output_dir)
elif model_name in BERT_LIST:
    logging.info(f"Collecting hs for model {model_name} in MASKED mode.")
    collect_hs_masked(dl, model_name, model, tokenizer, 
        tokenizer.mask_token_id, V, output_dir)
else:
    raise ValueError(f"Model name {model_name} incorrect.")
    


