#%%
import warnings
import logging
import os
import sys
import argparse
import coloredlogs

import numpy as np
import pickle
import torch
from tqdm import tqdm, trange
import random
import pandas as pd
import time

import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import wandb

from rlace import solve_adv_game, solve_adv_game_param_free, \
    init_classifier, get_majority_acc, solve_adv_game_param_free_twoPs
import algorithms.inlp.debias
from classifiers.classifiers import BinaryParamFreeClf
from eval_helpers import full_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
else: 
    torch.device("cpu")
    logging.warning("No GPU found")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Compute H Matrices')
    argparser.add_argument(
        "-outdir",
        type=str,
        help="Directory for exporting run eval"
    )
    argparser.add_argument(
        "-k",
        type=int,
        help="Rank of "
    )
    argparser.add_argument(
        "-niter",
        type=int,
        help="Number of iterations of RLACE"
    )
    argparser.add_argument(
        "-nruns",
        type=int,
        help="Number of runs of the experiment"
    )
    argparser.add_argument(
        "-seed",
        type=int,
        help="Seed for shuffling data"
    )
    parser.add_argument(
        '-wbn', 
        dest='wandb_name', 
        default="", 
        type=str, 
        help="Name of wandb run."
    )
    return argparser.parse_args()

args = get_args()
logging.info(args)

RANK = args.k
#RANK = 1
RLACE_NITER = args.niter
#RLACE_NITER = 1000
NRUNS = 5
#NRUNS = args.nruns
SEED = args.seed
#SEED = 0
DIRECTORY = args.outdir
#DIRECTORY = "real_runs_18"
WBN = args.wandb_name

#%% 
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "linzen"
DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}.pkl"
OUTPUT_DIR = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/{DIRECTORY}/"

assert os.path.exists(OUTPUT_DIR), \
    f"Output dir doesn't exist: {OUTPUT_DIR}"

DIAG_RLACE_U_OUTDIR = os.path.join(OUTPUT_DIR, "diag_rlace_u")

assert os.path.exists(DIAG_RLACE_U_OUTDIR), \
    f"Torch output dir doesn't exist: {DIAG_RLACE_U_OUTDIR}"

logging.info(
    f"Running: rank {RANK} niter {RLACE_NITER}"
)

TRAIN_OBS = 30000
VAL_OBS = 10000
TEST_OBS = 20000

run_args = {
    "rank": RANK,
    "rlace_niter": RLACE_NITER,
    "nruns": NRUNS,
    "seed": SEED,
    "out_dir": OUTPUT_DIR,
    "model_name": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "train_obs": TRAIN_OBS,
    "val_obs": VAL_OBS,
    "test_obs": TEST_OBS
}

if WBN:
    wandb.init(
        project="usagebasedprobing", 
        entity="cguerner",
        name=WBN
    )
    wandb.config.update(run_args)
    WB = True
else:
    WB = False

#%%
with open(DATASET, 'rb') as f:      
    data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])

#%%
X = np.array([x for x in data["h"]])
U = np.array([x for x in data["u"]])
y = np.array([yi for yi in data["y"]])
del data
np.random.seed(SEED)

#%%
for i in trange(NRUNS):
        
    #%%
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)

    train_lastind = TRAIN_OBS
    val_lastind = train_lastind + VAL_OBS
    test_lastind = val_lastind + TEST_OBS
    X_train, X_val, X_test = X[idx[:train_lastind]], X[idx[train_lastind:val_lastind]], X[idx[val_lastind:test_lastind]]
    U_train, U_val, U_test = U[idx[:train_lastind]], U[idx[train_lastind:val_lastind]], U[idx[val_lastind:test_lastind]]
    y_train, y_val, y_test = y[idx[:train_lastind]], y[idx[train_lastind:val_lastind]], y[idx[val_lastind:test_lastind]]

    #%%
    rlace_optimizer_class = torch.optim.SGD
    rlace_optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
    rlace_optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
    rlace_epsilon = 0.001 # stop 0.1% from majority acc
    rlace_batch_size = 256
    dim = X_train.shape[1]

    #%%
    logging.info("Training RLACE with diagnostic data")
    start = time.time()
    
    diag_rlace_u_outfile = os.path.join(DIAG_RLACE_U_OUTDIR, f"rk_{RANK}_niter_{RLACE_NITER}_{i}.pt")
    
    diag_rlace_output = solve_adv_game(
        X_train, y_train, X_val, y_val, rank=RANK, device=device, 
        out_iters=RLACE_NITER, optimizer_class=rlace_optimizer_class, 
        optimizer_params_P =rlace_optimizer_params_P, 
        optimizer_params_predictor=rlace_optimizer_params_predictor, 
        epsilon=rlace_epsilon,batch_size=rlace_batch_size,
        torch_outfile=diag_rlace_u_outfile, wb=WB
    )
    end = time.time()
    
    diag_rlace_results = full_eval(
        diag_rlace_output, 
        X_train, U_train, y_train, 
        X_val, U_val, y_val,
        X_test, U_test, y_test, end-start
    )


    """
    #%%
    #versions = ["original", "positively_functional", "negatively_functional"]
    #functional_rlace_results = {}
    #for version in versions:
    #    logging.info(f"Training functional RLACE version: {version} ")
    #    start = time.time()
    #    functional_rlace_output = solve_adv_game_param_free(
    #        X_train, U_train, y_train, X_val, U_val, y_val, version,
    #        rank=RANK, device=device, 
    #        out_iters=RLACE_NITER, optimizer_class=rlace_optimizer_class, 
    #        optimizer_params_P =rlace_optimizer_params_P, 
    #        epsilon=rlace_epsilon,batch_size=rlace_batch_size
    #    )
    #    end = time.time()
    #    functional_rlace_results[version] = full_eval(
    #        functional_rlace_output, 
    #        X_train, U_train, y_train, 
    #        X_val, U_val, y_val,
    #        X_test, U_test, y_test,
    #        end - start
    #    )

    #%%
    logging.info("Training INLP with diagnostic data")
    num_classifiers = RANK
    classifier_class = SGDClassifier #Perceptron
    input_dim = X_train.shape[1]
    is_autoregressive = True
    min_accuracy = 0.0

    start = time.time()
    diag_inlp_output = algorithms.inlp.debias.get_debiasing_projection(
        classifier_class, {}, num_classifiers, input_dim, is_autoregressive, 
        min_accuracy, X_train, y_train, X_val, y_val, by_class = False
    )
    end = time.time()

    inlp_results = full_eval(
        diag_inlp_output, 
        X_train, U_train, y_train, 
        X_test, U_test, y_test,
        end - start
    )
    #I = np.eye(P.shape[0])
    #P_alternative = I - np.sum(rowspace_projections, axis = 0)
    #P_by_product = I.copy()

    #for P_Rwi in rowspace_projections:

    #    P_Nwi = I - P_Rwi
    #    P_by_product = P_Nwi.dot(P_by_product)
    """

    #%%
    full_results = dict(
        run = i,
        run_args = run_args,
        diag_rlace = diag_rlace_results,
        #functional_rlace = functional_rlace_results,
        #inlp = inlp_results,
        maj_acc_test = get_majority_acc(y_test),
        maj_acc_val = get_majority_acc(y_val),
        maj_acc_train = get_majority_acc(y_train)
    )
    
    #%%
    outfile_path = os.path.join(OUTPUT_DIR, f"run_k_{RANK}_n_{RLACE_NITER}_{i}_{NRUNS}.pkl")

    with open(outfile_path, 'wb') as f:
        pickle.dump(full_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported {outfile_path}")


logging.info("Done")
