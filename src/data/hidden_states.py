#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

from transformers import BertTokenizerFast, BertForMaskedLM
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

sys.path.append('./src/')

from data.utils import get_dataset

import ipdb

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% TROUBLESHOOTING VERSION
#SEED = 0
#CHKPT = "2000k"
#DATASET = "wikipedia" # "wikipedia", "bookcorpus"
#NSAMPLES = 100
#BATCH_SIZE = 8

#%%
#if CHKPT == "none":
#    MODEL_NAME = f"google/multiberts-seed_{SEED}"
#else:
#    MODEL_NAME = f"google/multiberts-seed_{SEED}-step_{CHKPT}"

#TOKENIZER = BertTokenizerFast.from_pretrained(
#    MODEL_NAME, model_max_length=512
#)
#MODEL = BertForMaskedLM.from_pretrained(
#    MODEL_NAME, 
#    cache_dir="/cluster/scratch/cguerner/thesis_data/hf_cache", 
#    is_decoder=False
#)

#%%
#MAIN_HIDDEN_STATES_DIR = (
#    "/cluster/scratch/cguerner/thesis_data/hidden_states/multiberts/")

#assert os.path.exists(MAIN_HIDDEN_STATES_DIR), \
#        "Multiberts directory doesn't exist"

#EXPORT_DIR = os.path.join(
#    MAIN_HIDDEN_STATES_DIR, f"{SEED}/{CHKPT}/{DATASET}"
#)

#if not os.path.exists(EXPORT_DIR):
#    os.makedirs(EXPORT_DIR)
#    logging.info(f"Created directory: {EXPORT_DIR}")

# Create dir specifically for this run
#RUN_OUTPUT_DIR = os.path.join(EXPORT_DIR, TIMESTAMP)
#assert not os.path.exists(RUN_OUTPUT_DIR), "Run output dir already exists"
#os.mkdir(RUN_OUTPUT_DIR)
#logging.info(f"Created directory: {RUN_OUTPUT_DIR}")

#logging.info(f"{MODEL_NAME} - exporting hidden states from {DATASET} using "
#             f"{str(NSAMPLES)} into {RUN_OUTPUT_DIR}.")

#%%
"""
subds = ds["train"].train_test_split(train_size=NSAMPLES, shuffle=True)\
    ["train"]

def encode(examples):
    encoded_examples = TOKENIZER(
        examples["text"], return_tensors='pt', 
        padding="max_length", truncation=True
    )
    return encoded_examples

subds = subds.map(encode, batched=True)

#%%
# TODO: how to include id??
subds.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask"]
)

#%%
#TODO: optimize batch size
sample_dataloader = DataLoader(
    subds, 
    batch_size = 8, 
    pin_memory=True
)
"""
#%% Helpers
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
    for j in range(n_samples):
        sample = batch_input_ids[j,:]
        mask = batch_attention_mask[j,:]
        masked_sample, mask_token_index = mask_random_token(
            sample, mask, mask_token_id
        )
        batch_masked_ids.append(masked_sample)
        batch_masked_token_indices.append(mask_token_index)

    batch_masked_input_ids = torch.stack(batch_masked_ids, axis=0)
    return batch_masked_input_ids, batch_masked_token_indices


"""
def export_batch_data(batch_num, batch_data, batch_output):
    export_path = os.path.join(RUN_OUTPUT_DIR, f"batch_{batch_num}.npz")
    lmhead_hidden_states = output["hidden_states"][1].cpu().numpy()
    last_encoder_hidden_states = output["hidden_states"][0][-1].cpu().numpy()
    np.savez(
        export_path, 
        #id=batch_data["id"],
        input_ids=batch_data["input_ids"].detach().cpu().numpy(),
        token_type_ids=batch_data["token_type_ids"].detach().cpu().numpy(),
        attention_mask=batch_data["attention_mask"].detach().cpu().numpy(),
        #logits=batch_output["logits"].detach().cpu().numpy(),
        lmhead_hidden_states=lmhead_hidden_states,
        last_encoder_hidden_states=last_encoder_hidden_states
    )
"""

def get_masked_hs(batch_mlm_hs, batch_masked_token_indices):
    masked_hs = []
    for k, masked_index in enumerate(batch_masked_token_indices):
        masked_hs.append(batch_mlm_hs[k,masked_index,:])
    return torch.stack(masked_hs,axis=0).cpu().detach().numpy() 


def export_batch_mlm_hs(output_dir, batch_num, mlm_hs):
    export_path = os.path.join(output_dir, f"batch_{batch_num}.npy")
    np.save(export_path, mlm_hs)


#%%
def collect_hidden_states(device,
                          model, 
                          tokenizer, 
                          dataset, 
                          nsamples, 
                          output_dir, 
                          batch_size=8):
    
    model = model.to(device)
    mask_token_id = tokenizer.mask_token_id

    ds = get_dataset(dataset)

    sample_dataloader = DataLoader(
        ds["train"], 
        batch_size = batch_size, 
        #pin_memory=True,
        shuffle=True
    )

    #TODO: fix the progress bar display to show actual number
    # of batches, not the count of batches for whole dataset
    for i, batch in enumerate(pbar:=tqdm(sample_dataloader)):
        pbar.set_description(f"Generating hidden states")

        tokenized_text = tokenizer(batch["text"], 
            add_special_tokens=False,
            return_special_tokens_mask=False,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )

        masked_input_ids, masked_token_indices = mask_batch(
            tokenized_text["input_ids"], 
            tokenized_text["attention_mask"],
            mask_token_id
        )

        masked_input_ids = masked_input_ids.to(device)
        token_type_ids = tokenized_text["token_type_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

        with torch.no_grad():
            output = model(
                input_ids=masked_input_ids, token_type_ids=token_type_ids, 
                attention_mask=attention_mask, output_hidden_states=True
            )
            mlm_hs = model.cls.predictions.transform(
                output["hidden_states"][-1]
            )

        masked_lm_hs = get_masked_hs(mlm_hs, masked_token_indices)

        export_batch_mlm_hs(output_dir, i, masked_lm_hs)
        torch.cuda.empty_cache()

        if (i+1) * batch_size > nsamples:
            break

    logging.info(f"Finished exporting data to {output_dir}")

#%% METHODS FOR RUNNING THIS FILE AS A SCRIPT
def get_args():
    argparser = argparse.ArgumentParser(description='Collect hidden states')
    argparser.add_argument(
        "--seed",
        type=int,
        help="MultiBERTs seed for tokenizer and model"
    )
    argparser.add_argument(
        "--chkpt",
        type=str,
        choices=["0k", "20k", "40k", "60k", "80k", "100k", "120k", "140k", 
            "160k", "180k", "200k", "300k", "400k", "500k", "600k", "700k", 
            "800k", "900k", "1000k", "1100k", "1200k", "1300k", "1400k", 
            "1500k", "1600k", "1700k", "1800k", "1900k", "2000k", "none"],
        help="MultiBERTs checkpoint for tokenizer and model"
    )
    argparser.add_argument(
        "--dataset", 
        type=str,
        choices=["wikipedia", "bookcorpus"],
        required=True,
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "--nsamples",
        type=int,
        help="Number of hidden states to compute"
    )
    return argparser.parse_args()

def main():
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
    else: 
        torch.device("cpu")
        logging.warning("No GPU found")

    args = get_args()
    logging.info(args)

    SEED = args.seed
    CHKPT = args.chkpt
    DATASET = args.dataset # "wikipedia", "bookcorpus"
    NSAMPLES = args.nsamples
    BATCH_SIZE = 8

    if CHKPT == "none":
        MODEL_NAME = f"google/multiberts-seed_{SEED}"
    else:
        MODEL_NAME = f"google/multiberts-seed_{SEED}-step_{CHKPT}"

    TOKENIZER = BertTokenizerFast.from_pretrained(
        MODEL_NAME, model_max_length=512
    )
    MODEL = BertForMaskedLM.from_pretrained(
        MODEL_NAME, 
        cache_dir="/cluster/scratch/cguerner/thesis_data/hf_cache", 
        is_decoder=False
    )

    MAIN_HIDDEN_STATES_DIR = (
        "/cluster/scratch/cguerner/thesis_data/hidden_states/multiberts/")

    assert os.path.exists(MAIN_HIDDEN_STATES_DIR), \
            "Multiberts directory doesn't exist"

    EXPORT_DIR = os.path.join(
        MAIN_HIDDEN_STATES_DIR, f"{SEED}/{CHKPT}/{DATASET}"
    )

    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        logging.info(f"Created directory: {EXPORT_DIR}")

    # Create dir specifically for this run
    RUN_OUTPUT_DIR = os.path.join(EXPORT_DIR, TIMESTAMP)
    assert not os.path.exists(RUN_OUTPUT_DIR), "Run output dir already exists"
    os.mkdir(RUN_OUTPUT_DIR)
    logging.info(f"Created directory: {RUN_OUTPUT_DIR}")

    logging.info(
        f"{MODEL_NAME} - exporting hidden states from {DATASET} using "
        f"{str(NSAMPLES)} into {RUN_OUTPUT_DIR}.")
    
    collect_hidden_states(
        device,
        MODEL,
        TOKENIZER,
        DATASET,
        NSAMPLES,
        RUN_OUTPUT_DIR,
        BATCH_SIZE
    )

#%%
if __name__=="__main__":
    main()