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
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers.activations import gelu

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC

#sys.path.append('..')
sys.path.append('./src/')

from paths import OUT, HF_CACHE, FR_DATASETS
from utils.cuda_loaders import get_device
from utils.lm_loaders import get_model, get_tokenizer, get_V, \
    GPT2_LIST, BERT_LIST, SUPPORTED_AR_MODELS
from utils.dataset_loaders import load_preprocessed_dataset
import ipdb

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# General Helpers  #
####################
class CustomDataset(Dataset, ABC):
    def __init__(self, list_of_samples):
        self.data = list_of_samples
        self.n_instances = len(self.data)
        
    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.data[index]

def export_batch(output_dir, batch_num, batch_data):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
    
    with open(export_path, 'wb') as f:
        pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%#################
# AR Helpers       #
####################
def remove_bos(input_ids):
    first_token = [x[0] for x in input_ids]
    unique_tokens = np.unique(first_token)
    if (unique_tokens == np.array([1])).all():
        return [x[1:] for x in input_ids]
    else:
        return input_ids

def batch_tokenize_tgts(model_name, tgts, tokenizer):
    if model_name in GPT2_LIST:
        return tokenizer(tgts)["input_ids"]
    elif model_name == "llama2": 
        return remove_bos(tokenizer(tgts)["input_ids"])
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

def batch_tokenize_text(text, tokenizer):
    return tokenizer(
        text, return_tensors='pt', 
        padding="max_length", truncation=True
    )

def get_raw_sample_hs(model_name, pre_tgt_ids, attention_mask, tgt_ids, model):
    nopad_ti = pre_tgt_ids[attention_mask == 1]
    tgt_ti = torch.cat((nopad_ti, torch.LongTensor(tgt_ids)), 0)
    if model_name == "llama2":
        tgt_ti = tgt_ti.unsqueeze(0)
    tgt_ti_dev = tgt_ti.to(device)
    with torch.no_grad():
        #TODO: check that the logits are the same
        output = model(
            input_ids=tgt_ti_dev, 
            #attention_mask=attention_mask, 
            labels=tgt_ti_dev,
            output_hidden_states=True
        )
    if model_name == "llama2":
        return tgt_ti.numpy(), output.hidden_states[-1][0].cpu().numpy()
    elif model_name in GPT2_LIST:
        return tgt_ti.numpy(), output.hidden_states[-1].cpu().numpy()
    else:
        raise NotImplementedError(f"Model {model_name} not yet implemented")


def get_tgt_hs(raw_hs, tgt_tokens):
    #pre_verb_hs = raw_hs[:-(len(tgt_tokens) + 1)]
    #verb_hs = raw_hs[-(tgt_tokens.shape[0]+1):]
    #ipdb.set_trace()
    tgt_hs = raw_hs[-(len(tgt_tokens)+1):]
    return tgt_hs

def get_batch_hs_ar(model_name, batch, model, tokenizer, V):
    batch_hs = []
    tok_facts = batch_tokenize_tgts(model_name, batch["fact"], tokenizer)
    tok_foils = batch_tokenize_tgts(model_name, batch["foil"], tokenizer)
    tok_text = batch_tokenize_text(batch["pre_tgt_text"], tokenizer)

    #TODO: this sample by sample business is not really necessary now, should batch it
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
        fact_ti, fact_raw_hs = get_raw_sample_hs(model_name, ti, am, tfa, model)
        fact_hs = get_tgt_hs(fact_raw_hs, tfa)

        foil_ti, foil_raw_hs = get_raw_sample_hs(model_name, ti, am, tfo, model)
        foil_hs = get_tgt_hs(foil_raw_hs, tfo)

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

def collect_hs_ar(model_name, dl, model, tokenizer, V, output_dir):
    for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

        batch_data = get_batch_hs_ar(model_name, batch, model, tokenizer, V)
        export_batch(output_dir, i, batch_data)

        torch.cuda.empty_cache()
    logging.info(f"Finished exporting data to {output_dir}.")

#%%#################
# Masked Helpers   #
####################
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

        fact_tok_ids = tokenizer.encode(fact)[1:-1]
        foil_tok_ids = tokenizer.encode(foil)[1:-1]

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

def collect_hs_masked(dl, model_name, model, tokenizer, mask_token_id, V, output_dir):
    for i, batch in enumerate(pbar:=tqdm(dl)):
        pbar.set_description(f"Generating hidden states")

        # Compute MASK hidden state
        tokenized_text = tokenizer(
            batch["masked"], return_tensors='pt', 
            padding="max_length", truncation=True
        )
        #ipdb.set_trace()
        batch_mask_indices = torch.where(
            (tokenized_text["input_ids"] == mask_token_id)
        )
        input_ids = tokenized_text["input_ids"].to(device)
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

        #TODO:sanity check this with prob dist
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
        "-dataset", 
        type=str,
        choices=["linzen", "CEBaB"] + FR_DATASETS,
        default="linzen",
        help="Dataset to collect hidden states for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-split",
        type=str,
        choices=["train", "dev", "test"],
        default=None,
        help="For UD data, specifies which split to collect hs for"
    )
    return argparser.parse_args()


if __name__=="__main__":
    args = get_args()
    logging.info(args)

    dataset_name = args.dataset
    model_name = args.model
    split = args.split
    #dataset_name = "CEBaB"
    #model_name = "gpt2-large"
    #split = "train"
    batch_size = 64

    # Load model, tokenizer
    device = get_device()

    tokenizer = get_tokenizer(model_name)
    #pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    model = get_model(model_name)
    V = get_V(model_name, model)

    model = model.to(device)

    # load data
    data = load_preprocessed_dataset(dataset_name, model_name, split)
    ds = CustomDataset(data)
    dl = DataLoader(dataset = ds, batch_size=batch_size)

    # Output dir
    if split is None:
        output_dir = os.path.join(OUT, f"hidden_states/{dataset_name}/{model_name}")
    else:
        output_dir = os.path.join(OUT, f"hidden_states/{dataset_name}/{model_name}/{split}")

    os.makedirs(output_dir, exist_ok=False)

    # Collect HS
    if model_name in SUPPORTED_AR_MODELS:
        logging.info(f"Collecting hs for model {model_name} in AR mode.")
        collect_hs_ar(model_name, dl, model, tokenizer, V, output_dir)
    elif model_name in BERT_LIST:
        logging.info(f"Collecting hs for model {model_name} in MASKED mode.")
        collect_hs_masked(dl, model_name, model, tokenizer, 
            tokenizer.mask_token_id, V, output_dir)
    else:
        raise ValueError(f"Model name {model_name} incorrect.")
    


