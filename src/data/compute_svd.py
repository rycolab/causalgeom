#%%
import warnings
import logging
import os
import coloredlogs
import argparse

import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%% sample H mat
def get_hmat_sample(hmat, nsamples):
    """ h should be a matrix with dimensions Nx768"""
    index = np.random.choice(hmat.shape[0], nsamples, replace=False)
    return hmat[index,:]

#%%
def export_svd(h, output_filepath):
    """ h should be a matrix with dimensions Nx768"""
    u, s, vh = np.linalg.svd(h.T)
    del vh

    out = {"u": u, "s": s}
    with open(output_filepath, 'wb') as outfile:
        pickle.dump(out, outfile, pickle.HIGHEST_PROTOCOL)

    logging.info(f"Full SVD computed and saved.")

#%% 
#h = np.concatenate([h_wiki, h_book], axis=0)

#%% Singular values
#out = np.linalg.svd(h_sub.T,compute_uv=False)
#svals_filepath = os.path.join(
#    OUTPUT_DIR, f"multiberts_{SEED}_{DATASET}_{DATETIME}_n{NSAMPLES}_svals.npy"
#)
#np.save(svals_filepath, out)

#logging.info(f"Sing values computed and saved: {pd.Series(out).describe()}")


#%% Right inverse
#pinv = np.linalg.pinv(h.T)

#pinv_filepath = os.path.join(
#    OUTPUT_DIR, 
#    f"multiberts_{SEED}_{DATASET}_{DATETIME}_n{NSAMPLES}_pinv.pickle"
#)
#with open(pinv_filepath, 'wb') as outfile:
#    pickle.dump(pinv, outfile, pickle.HIGHEST_PROTOCOL)

#logging.info(f"Pinv computed and saved")

#%% SCRIPT WORKFLOW; NOT TESTED
def get_args():
    argparser = argparse.ArgumentParser(description='ComputeTokenCounts')
    argparser.add_argument(
        "--dataset", 
        type=str,
        choices=["wikipedia", "bookcorpus"],
        required=True,
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "--seed",
        type=int,
        help="MultiBERTs seed for tokenizer and model"
    )
    argparser.add_argument(
        "--datetime",
        type=str,
        help="Datetime of hidden state export folder"
    )
    argparser.add_argument(
        "--nsamples",
        type=int,
        help="Number of samples for SVD"
    )
    return argparser.parse_args()

#%% 
def main():
    args = get_args()
    logging.info(args)

    DATASET = args.dataset # "wikipedia", "bookcorpus"
    SEED = args.seed
    DATETIME = args.datetime
    NSAMPLES = args.nsamples

    H_MATRIX_FOLDER = "/cluster/scratch/cguerner/thesis_data/h_matrices/"
    H_FILEPATH = os.path.join(H_MATRIX_FOLDER, 
        f"multiberts_{SEED}_{DATASET}_{DATETIME}.npy")

    assert os.path.exists(H_FILEPATH), \
        f"H matrix doesn't exist: {FILEDIR}"

    OUTPUT_DIR = "/cluster/scratch/cguerner/thesis_data/svd/"

    assert os.path.exists(OUTPUT_DIR), \
        f"SVD output dir doesn't exist: {OUTPUT_DIR}"

    logging.info(f"Saving SVD for H matrix multiberts/{SEED}/{DATASET}/{DATETIME}")

    # load H sample and export SVD
    h = np.load(H_FILEPATH)
    h = get_hmat_sample(h, NSAMPLES)

    svd_filepath = os.path.join(
        OUTPUT_DIR, 
        f"multiberts_{SEED}_{DATASET}_{DATETIME}_n{NSAMPLES}_svd.pickle"
    )
    export_svd(h, outfile)

#%%
if __name__=="__main__":
    main()