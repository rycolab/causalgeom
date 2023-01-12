#%%
import warnings
import logging
import os
import coloredlogs
import argparse

import numpy as np
from tqdm import tqdm
from zipfile import BadZipFile
import shutil

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%% TROUBLESHOOTING
#SEED = 0
#DATASET = "bookcorpus"
#DATETIME = "20221201_231959"
#DATASET = "wikipedia"
#DATETIME = "20221201_231956"
#FILEDIR = (f"/cluster/scratch/cguerner/thesis_data/hidden_states"
#            f"/multiberts/{SEED}/{DATASET}/{DATETIME}")
#OUTPUT_DIR = "/cluster/scratch/cguerner/thesis_data/h_matrices/"

#logging.info(f"Creating H matrix from {FILEDIR}")

#%%
def create_temp_files(batch_file_dir, tempdir, temp_nbatches=100):
    """ Loops through batch_file_dir batch npy files containing hidden states,
    concatenates them, and exports temporary files containing 100 batches each
    into tempdir
    """
    files = os.listdir(batch_file_dir)
    files.sort()

    j = 0
    tempcount = 0
    for i, filename in enumerate(tqdm(files)):
        filepath = os.path.join(batch_file_dir, filename)
        try:
            hidden_states = np.load(filepath)
        except BadZipFile:
            logging.warning(f"BadZipFile skipped {filename}")
            continue
        if j == 0:
            all_hidden_states = hidden_states
            j+=1
        elif j == (temp_nbatches-1) or i == len(files)-1:
            all_hidden_states = np.concatenate(
                (all_hidden_states, hidden_states), axis=0
            )
            tempfile = os.path.join(
                tempdir, 
                f"temp{tempcount}.npy"
            )
            np.save(tempfile, all_hidden_states)
            j=0
            tempcount+=1
        else:
            all_hidden_states = np.concatenate(
                (all_hidden_states, hidden_states), axis=0
            )
            j+=1

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
        hidden_states = np.load(filepath)
        all_temps.append(hidden_states)

    return np.concatenate(all_temps, axis=0)

#%%
def create_h_matrix(batch_file_dir, output_file, delete_batch_dir=False):
    """ Reads batch files containing hidden states and saves concatenated
    H matrix. Optionally deletes directory containing batch files.
    """

    OUTPUT_DIR = os.path.dirname(output_file)
    TEMPDIR = os.path.join(OUTPUT_DIR, "temp")
    os.mkdir(TEMPDIR)

    create_temp_files(batch_file_dir, TEMPDIR)
    h_matrix = concat_temp_files(TEMPDIR)

    np.save(output_file, h_matrix)

    logging.info(f"Exported H matrix to {output_file}")

    #%% Delete temp files
    shutil.rmtree(TEMPDIR)
    logging.info(f"Deleted tempdir: {TEMPDIR}")

    #%% Delete batch files
    if delete_batch_dir:
        shutil.rmtree(batch_file_dir)
        logging.info(f"Deleted batch dir: {batch_file_dir}")

#%% SCRIPT WORKFLOW (NOT TESTED)
def get_args():
    argparser = argparse.ArgumentParser(description='Compute H Matrices')
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
        "--datetime",
        type=str,
        help="Datetime of hidden state export folder"
    )
    return argparser.parse_args()


def main():
    args = get_args()
    logging.info(args)

    SEED = args.seed
    CHKPT = args.chkpt
    DATASET = args.dataset # "wikipedia", "bookcorpus"
    DATETIME = args.datetime

    FILEDIR = (f"/cluster/scratch/cguerner/thesis_data/hidden_states"
                f"/multiberts/{SEED}/{CHKPT}/{DATASET}/{DATETIME}")

    assert os.path.exists(FILEDIR), \
        f"Hidden state filedir doesn't exist: {FILEDIR}"

    OUTPUT_DIR = "/cluster/scratch/cguerner/thesis_data/h_matrices/"

    assert os.path.exists(OUTPUT_DIR), \
        f"H matrix output dir doesn't exist: {OUTPUT_DIR}"

    logging.info(
        f"Creating H matrix for multiberts/{SEED}/{CHKPT}/{DATASET}/{DATETIME}"
    )

    OUTFILE = os.path.join(
        OUTPUT_DIR, f"multiberts_{SEED}_{CHKPT}_{DATASET}_{DATETIME}.npy"
    )
    
    create_h_matrix(FILEDIR, OUTFILE)


if __name__=="__main__":
    main()