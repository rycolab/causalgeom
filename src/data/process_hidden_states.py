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


#%% Label creation
def format_sample(sample):
    hs, verb, iverb, verb_pos = sample
    if verb_pos == "VBZ":
        y = 0
        u = iverb - verb
    elif verb_pos == "VBP":
        y = 1
        u = verb - iverb
    else:
        raise ValueError(f"Unknown verb POS tag: {verb_pos}")
    return (hs, u, y)

def count_verb_tokens(sample):
    return max(
        len(sample["input_ids_verb"]), 
        len(sample["input_ids_iverb"])
    )

def format_sample_ar(sample):
    #TODO: debug into this
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

def format_batch(batch):
    formatted_batch = []
    count_drops = 0
    for sample in batch:
        formatted_sample = format_sample_ar(sample)
        if formatted_sample is None:
            count_drops+=1
        else:
            formatted_batch.append(formatted_sample)
        #formatted_batch.append(format_sample_shauli(sample))
    return formatted_batch, count_drops

def create_temp_files(batch_file_dir, tempdir, nbatches=None, temp_nbatches=100):
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
            batch_data, drop_count = format_batch(pickle.load(f))
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
def process_hidden_states(batch_file_dir, output_file, nbatches=None, delete_batch_dir=False):
    """ Reads batch files containing hidden states and saves concatenated
    H matrix. Optionally deletes directory containing batch files.
    """

    OUTPUT_DIR = os.path.dirname(output_file)
    TEMPDIR = os.path.join(OUTPUT_DIR, "temp")
    os.mkdir(TEMPDIR)

    create_temp_files(batch_file_dir, TEMPDIR, nbatches=nbatches)
    data = concat_temp_files(TEMPDIR)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported processed data to {output_file}")

    #%% Delete temp files
    shutil.rmtree(TEMPDIR)
    logging.info(f"Deleted tempdir: {TEMPDIR}")

    #%% Delete batch files
    if delete_batch_dir:
        shutil.rmtree(batch_file_dir)
        logging.info(f"Deleted batch dir: {batch_file_dir}")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "--dataset", 
        type=str,
        choices=["linzen"],
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "--model",
        type=str,
        choices=["bert-base-uncased", "gpt2"],
        help="MultiBERTs checkpoint for tokenizer and model"
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

    DATASET_NAME = "linzen"
    MODEL_NAME = "gpt2"
    
    logging.info(f"Creating processed dataset for dataset {DATASET_NAME}, model {MODEL_NAME}.")

    FILEDIR = (f"/cluster/work/cotterell/cguerner/usagebasedprobing/"
                f"out/hidden_states/{DATASET_NAME}/{MODEL_NAME}")

    assert os.path.exists(FILEDIR), \
        f"Hidden state filedir doesn't exist: {FILEDIR}"

    OUTFILE = (f"/cluster/work/cotterell/cguerner/usagebasedprobing/"
                f"datasets/processed/{DATASET_NAME}_{MODEL_NAME}_shauli.pkl")
    
    process_hidden_states(FILEDIR, OUTFILE, nbatches=args.nbatches)
