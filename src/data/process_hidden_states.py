#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import pickle
from tqdm import tqdm
import shutil

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% BATCH and SAMPLE processing

## HELPERS
def count_verb_tokens(sample):
    return max(
        len(sample["input_ids_verb"]), 
        len(sample["input_ids_iverb"])
    )

## MASKED 
def format_sample_masked(sample):
    hs = sample["hs"]
    verb = sample["verb_embedding"]
    iverb = sample["iverb_embedding"]
    verb_pos = sample["verb_pos"]
    max_tokens = count_verb_tokens(sample)
    if verb_pos == "VBZ" and max_tokens == 1:
        y = 0
        u = iverb - verb
        return (hs, u, y)
    elif verb_pos == "VBP" and max_tokens == 1:
        y = 1
        u = verb - iverb
        return (hs, u, y)
    elif max_tokens > 1:
        return None 
    else:
        raise ValueError(f"Unknown verb POS tag: {verb_pos}")

## AR
def format_sample_ar(sample):
    hs = sample["verb_hs"][0,:]
    verb = sample["verb_embedding"]
    iverb = sample["iverb_embedding"]
    verb_pos = sample["verb_pos"]
    max_tokens = count_verb_tokens(sample)
    if verb_pos == "VBZ" and max_tokens == 1:
        y = 0
        u = iverb.flatten() - verb.flatten()
        return (hs, u, y)
    elif verb_pos == "VBP" and max_tokens == 1:
        y = 1
        u = verb.flatten() - iverb.flatten()
        return (hs, u, y)
    elif max_tokens > 1:
        return None
    else:
        raise ValueError(f"Unknown verb POS tag: {verb_pos}")

## VERBS
def format_sample_verbs(sample):
    """ Returns verb and iverb input ids: (singular, plural)
    """
    verb_pos = sample["verb_pos"]
    verb = sample["verb"]
    iverb = sample["iverb"]
    id_verb = sample["input_ids_verb"]
    id_iverb = sample["input_ids_iverb"]
    if verb_pos == "VBZ":
        return (verb, id_verb, iverb, id_iverb)
    elif verb_pos == "VBP":
        return (iverb, id_iverb, verb, id_verb)
    else:
        raise ValueError(f"Unknown verb POS tag: {verb_pos}")

## HANDLER
def format_batch_handler(batch_data, out_type):
    formatted_batch = []
    count_drops = 0
    for sample in batch_data:
        if out_type == "ar": 
            formatted_sample = format_sample_ar(sample)
        elif out_type == "masked":
            formatted_sample = format_sample_masked(sample)
        else:
            formatted_sample = format_sample_verbs(sample)
        # NOTE: this is only needed for AR, mb adapt MASKED
        if formatted_sample is None:
            count_drops+=1
        else:
            formatted_batch.append(formatted_sample)
    return formatted_batch, count_drops


#%% FILE HANDLERS
def create_temp_files(batch_file_dir, tempdir, out_type, nbatches=None, temp_nbatches=100):
    """ Loops through batch_file_dir batch npy files containing hidden states,
    concatenates them, and exports temporary files containing 100 batches each
    into tempdir
    """
    files = os.listdir(batch_file_dir)
    files.sort()

    if nbatches is not None:
        files = files[:nbatches]
    
    total_drop_count = 0
    tempcount = 0
    data = []
    for i, filename in enumerate(tqdm(files)):
        filepath = os.path.join(batch_file_dir, filename)
        with open(filepath, 'rb') as f:      
            batch_data, drop_count = format_batch_handler(
                pickle.load(f), out_type
            )
            total_drop_count += drop_count
        if (i + 1) % temp_nbatches == 0 or i == len(files)-1:
            data = data + batch_data
            tempfile = os.path.join(
                tempdir, 
                f"temp{tempcount}.pkl"
            )
            with open(tempfile, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data = []
            tempcount+=1
        else:
            data = data + batch_data

    logging.info(
        f"Hidden state batches processed by creating {tempcount} tempfiles"
    )
    logging.info(
        "Number of samples dropped due to more"
        f" than one verb token: {total_drop_count}"
    )

def concat_temp_files(tempdir):
    """ Loops through tempdir files, concatenates them and returns full 
    H matrix
    """
    tempfiles = os.listdir(tempdir)

    all_temps = []
    for filename in tqdm(tempfiles):
        filepath = os.path.join(tempdir, filename)
        with open(filepath, 'rb') as f:      
            samples = pickle.load(f)
        all_temps = all_temps + samples

    return all_temps

#%%
def process_hidden_states(batch_file_dir, output_file, temp_dir, out_type, nbatches=None, delete_batch_dir=False):
    """ Reads batch files containing hidden states and saves concatenated
    H matrix. Optionally deletes directory containing batch files.
    """
    create_temp_files(batch_file_dir, temp_dir, out_type, nbatches=nbatches)
    data = concat_temp_files(temp_dir)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported processed data to {output_file}")

    #%% Delete temp files
    shutil.rmtree(temp_dir)
    logging.info(f"Deleted temp_dir: {temp_dir}")

    #%% Delete batch files
    if delete_batch_dir:
        shutil.rmtree(batch_file_dir)
        logging.info(f"Deleted batch dir: {batch_file_dir}")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "-dataset", 
        type=str,
        choices=["linzen"],
        default="linzen",
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=["bert-base-uncased", "gpt2"],
        help="MultiBERTs checkpoint for tokenizer and model"
    )
    argparser.add_argument(
        "-outtype",
        type=str,
        choices=["full", "verbs"],
        help="Export full dataset for probe training or just verbs"
    )
    argparser.add_argument(
        "-nbatches",
        type=int,
        required=False,
        default=None,
        help="Number of batches to process"
    )
    return argparser.parse_args()


if __name__=="__main__":
    args = get_args()
    logging.info(args)

    DATASET_NAME = args.dataset
    MODEL_NAME = args.model
    OUT_TYPE = args.outtype
    NBATCHES = args.nbatches
    #DATASET_NAME = "linzen"
    #MODEL_NAME = "gpt2"
    #OUT_TYPE = "full"
    #NBATCHES = 10

    if MODEL_NAME == "gpt2" and OUT_TYPE == "full":
        OUT_TYPE = "ar"
    elif MODEL_NAME == "bert-base-uncased" and OUT_TYPE == "full":
        OUT_TYPE = "masked"

    assert OUT_TYPE in ["ar", "masked", "verbs"], "Wrong outtype"

    logging.info(f"Creating {OUT_TYPE} dataset for raw data: {DATASET_NAME}, model {MODEL_NAME}.")

    FILEDIR = (f"/cluster/work/cotterell/cguerner/usagebasedprobing/"
                f"out/hidden_states/{DATASET_NAME}/{MODEL_NAME}")

    assert os.path.exists(FILEDIR), \
        f"Hidden state filedir doesn't exist: {FILEDIR}"

    OUTFILE = (f"/cluster/work/cotterell/cguerner/usagebasedprobing/"
                f"datasets/processed/{DATASET_NAME}_{MODEL_NAME}_{OUT_TYPE}.pkl")
    
    assert not os.path.isfile(OUTFILE), \
        f"Output file {OUTFILE} already exists"

    OUTPUT_DIR = os.path.dirname(OUTFILE)
    TEMPDIR = os.path.join(OUTPUT_DIR, f"temp_{MODEL_NAME}_{OUT_TYPE}")
    os.mkdir(TEMPDIR)

    process_hidden_states(FILEDIR, OUTFILE, TEMPDIR, OUT_TYPE, nbatches=NBATCHES)
