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
import torch

#sys.path.append('..')
sys.path.append('./src/')

from paths import OUT, DATASETS, FR_DATASETS
from utils.lm_loaders import GPT2_LIST, BERT_LIST

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%% FILE HANDLERS
def create_temp_files(batch_file_dir, tempdir, nbatches=None, temp_nbatches=100):
    """ Loops through batch_file_dir batch npy files containing hidden states,
    concatenates them, and exports temporary files containing 100 batches each
    into tempdir
    """
    files = os.listdir(batch_file_dir)
    files.sort()

    if nbatches is not None:
        files = files[:nbatches]
    
    #total_drop_count = 0
    tempcount = 0
    data = []
    for i, filename in enumerate(tqdm(files)):
        filepath = os.path.join(batch_file_dir, filename)
        with open(filepath, 'rb') as f:      
            batch_data = pickle.load(f)
        data.append(batch_data)
        if (i + 1) % temp_nbatches == 0 or i == len(files)-1:
            data = np.vstack(data)
            tempfile = os.path.join(
                tempdir, 
                f"temp{tempcount}.pkl"
            )
            with open(tempfile, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data = []
            tempcount+=1

    logging.info(
        f"Hidden state batches processed by creating {tempcount} tempfiles"
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
        all_temps.append(samples)

    return np.vstack(all_temps)

#%%
def process_hidden_states(batch_file_dir, output_file, temp_dir, nbatches=None, delete_batch_dir=False):
    create_temp_files(batch_file_dir, temp_dir, nbatches=nbatches)
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


if __name__=="__main__":
    args = get_args()
    logging.info(args)

    language = args.language
    model_name = args.model
    #language = "en"
    #model_name = "bert-base-uncased"

    logging.info(
        f"Creating other hidden states dataset for language "
        f"{language}, model {model_name}."
    )

    FILEDIR = os.path.join(OUT, f"hidden_states/{language}/{model_name}")
    OUTFILE = os.path.join(DATASETS, f"processed/{language}/other_hidden_states/{model_name}.pkl")
    OUTPUT_DIR = os.path.dirname(OUTFILE)
    TEMPDIR = os.path.join(OUTPUT_DIR, f"temp_{model_name}")    
        
    assert os.path.exists(FILEDIR), \
        f"Hidden state filedir doesn't exist: {FILEDIR}"
    
    assert not os.path.exists(TEMPDIR), f"Temp dir {TEMPDIR} already exists"
    os.makedirs(TEMPDIR)

    process_hidden_states(FILEDIR, OUTFILE, TEMPDIR)
