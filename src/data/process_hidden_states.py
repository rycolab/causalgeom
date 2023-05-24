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

#sys.path.append('..')
sys.path.append('./src/')

from paths import OUT, DATASETS, FR_DATASETS
from utils.lm_loaders import GPT2_LIST, BERT_LIST

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% BATCH and SAMPLE processing

## HELPERS
def count_tgt_tokens(sample):
    return max(
        len(sample["input_ids_fact"]), 
        len(sample["input_ids_foil"])
    )

def define_target(tgt_label):
    if (tgt_label == "VBZ" or tgt_label == "Masc"):
        return 0
    elif (tgt_label == "VBP" or tgt_label == "Fem"):
        return 1
    else:
        raise ValueError(f"Incorrect tgt label {tgt_label}")

## MASKED 
def format_sample_masked(sample):
    hs = sample["hs"]
    fact_emb = sample["fact_embedding"]
    foil_emb = sample["foil_embedding"]
    fact = sample["fact"]
    foil = sample["foil"]
    tgt_label = define_target(sample["tgt_label"])
    max_tokens = count_tgt_tokens(sample)
    if tgt_label == 0 and max_tokens == 1:
        y = 0
        u = foil - fact_emb
        return (hs, u, y, fact, foil)
    elif tgt_label == 1 and max_tokens == 1:
        y = 1
        u = fact_emb - foil
        return (hs, u, y, fact, foil)
    else: # max_tokens > 1:
        return None 
    
## AR
def format_sample_ar(sample):
    hs = sample["fact_hs"][0,:]
    fact_emb = sample["fact_embedding"]
    foil_emb = sample["foil_embedding"]
    fact = sample["fact"]
    foil = sample["foil"]
    tgt_label = define_target(sample["tgt_label"])
    max_tokens = count_tgt_tokens(sample)
    if tgt_label == 0 and max_tokens == 1:
        y = 0
        u = foil_emb.flatten() - fact_emb.flatten()
        return (hs, u, y, fact, foil)
    elif tgt_label == 1 and max_tokens == 1:
        y = 1
        u = fact_emb.flatten() - foil_emb.flatten()
        return (hs, u, y, fact, foil)
    else: # max_tokens > 1
        return None
    
## VERBS
def format_sample_tgt(sample):
    """ Returns fact and foil input ids: 
        (singular, plural), (Masc, Fem)
    """
    tgt_label = define_target(sample["tgt_label"])
    fact = sample["fact"]
    foil = sample["foil"]
    id_fact = sample["input_ids_fact"]
    id_foil = sample["input_ids_foil"]
    if tgt_label == 0:
        return (fact, id_fact, foil, id_foil, fact_pos)
    else:
        return (foil, id_foil, fact, id_fact, fact_pos)
    
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
            formatted_sample = format_sample_tgt(sample)
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
        choices=["linzen"] + FR_DATASETS,
        default="linzen",
        help="Dataset to process hidden states for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=BERT_LIST + GPT2_LIST,
        help="Model to process hidden states for"
    )
    argparser.add_argument(
        "-outtype",
        type=str,
        choices=["full", "tgt"],
        help="Export full dataset for probe training or just tgt"
    )
    argparser.add_argument(
        "-nbatches",
        type=int,
        required=False,
        default=None,
        help="Number of batches to process"
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

    DATASET_NAME = args.dataset
    MODEL_NAME = args.model
    OUT_TYPE = args.outtype
    NBATCHES = args.nbatches
    SPLIT = args.split
    #DATASET_NAME = "linzen"
    #MODEL_NAME = "gpt2-medium"
    #OUT_TYPE = "full"
    #NBATCHES = 10
    #SPLIT = "train"

    if MODEL_NAME in GPT2_LIST and OUT_TYPE == "full":
        OUT_TYPE = "ar"
    elif MODEL_NAME in BERT_LIST and OUT_TYPE == "full":
        OUT_TYPE = "masked"

    assert OUT_TYPE in ["ar", "masked", "tgt"], "Wrong outtype"

    logging.info(
        f"Creating {OUT_TYPE} dataset for raw data: "
        f"{DATASET_NAME}, model {MODEL_NAME}."
    )

    if SPLIT is None:
        FILEDIR = os.path.join(OUT, f"hidden_states/{DATASET_NAME}/{MODEL_NAME}")
        OUTFILE = os.path.join(DATASETS, f"processed/{DATASET_NAME}/{OUT_TYPE}/{DATASET_NAME}_{MODEL_NAME}_{OUT_TYPE}.pkl")
        OUTPUT_DIR = os.path.dirname(OUTFILE)
        TEMPDIR = os.path.join(OUTPUT_DIR, f"temp_{MODEL_NAME}_{OUT_TYPE}")
    else:
        FILEDIR = os.path.join(OUT, f"hidden_states/{DATASET_NAME}/{MODEL_NAME}/{SPLIT}")
        OUTFILE = os.path.join(DATASETS, f"processed/{DATASET_NAME}/{OUT_TYPE}/{DATASET_NAME}_{MODEL_NAME}_{OUT_TYPE}_{SPLIT}.pkl")
        OUTPUT_DIR = os.path.dirname(OUTFILE)
        TEMPDIR = os.path.join(OUTPUT_DIR, f"temp_{MODEL_NAME}_{OUT_TYPE}_{SPLIT}")
    
        
    assert os.path.exists(FILEDIR), \
        f"Hidden state filedir doesn't exist: {FILEDIR}"
    
    #assert not os.path.isfile(OUTFILE), \
    #    f"Output file {OUTFILE} already exists"

    
    assert not os.path.exists(TEMPDIR), f"Temp dir {TEMPDIR} already exists"
    os.makedirs(TEMPDIR)

    process_hidden_states(FILEDIR, OUTFILE, TEMPDIR, OUT_TYPE, nbatches=NBATCHES)
