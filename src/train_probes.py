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

#from rlace import solve_adv_game, solve_adv_game_param_free, \
#    init_classifier, get_majority_acc, solve_adv_game_param_free_twoPs
from algorithms.rlace.rlace import solve_adv_game, init_classifier, get_majority_acc

import algorithms.inlp.debias
from classifiers.classifiers import BinaryParamFreeClf
from utils.cuda_loaders import get_device
from evals.kl_eval import compute_kls, load_model_eval
from evals.usage_eval import full_usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

device = get_device()

#%%#################
# Args             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Compute H Matrices')
    argparser.add_argument(
        "-outdir",
        type=str,
        help="Directory for exporting run eval"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=["gpt2", "bert-base-uncased"],
        help="Model used to extract hidden states & embeddings"
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
        "-plr",
        type=float,
        default=0.003,
        help="Learning rate for P" 
    )
    argparser.add_argument(
        "-n_lr_red",
        type=int,
        default=5,
        help="Number of ReduceLROnPlateau reductions" 
    )
    argparser.add_argument(
        "-nruns",
        type=int,
        default=3,
        help="Number of runs of the experiment"
    )
    argparser.add_argument(
        "-train_obs",
        type=int,
        default=30000,
        help="Number of train obs"
    )
    argparser.add_argument(
        "-seed",
        type=int,
        help="Seed for shuffling data"
    )
    argparser.add_argument(
        '-wbn', 
        dest='wandb_name', 
        default="", 
        type=str, 
        help="Name of wandb run."
    )
    return argparser.parse_args()

MODE = "job" # "debug"

if MODE == "job":
    args = get_args()
    logging.info(args)

    DATASET_NAME = "linzen"
    MODEL_NAME = args.model
    RANK = args.k
    RLACE_NITER = args.niter
    PLR = args.plr
    
    #scheduler
    SCHED_NRED = args.n_lr_red
    SCHED_FACTOR = .5
    SCHED_PATIENCE = 4

    #data
    NRUNS = args.nruns
    TRAIN_OBS = args.train_obs
    VAL_OBS = 10000
    TEST_OBS = 20000
    SEED = args.seed

    OUTPUT_FOLDER = args.outdir
    WBN = args.wandb_name
else:
    logging.warn("RUNNING IN DEBUG MODE.")
    DATASET_NAME = "linzen"
    MODEL_NAME = "gpt2" #"bert-base-uncased"
    RANK = 1
    RLACE_NITER = 1000
    PLR=0.003

    #scheduler
    SCHED_NRED = 5
    SCHED_FACTOR = .5
    SCHED_PATIENCE = 4

    #data
    NRUNS = 1
    TRAIN_OBS = 60000
    VAL_OBS = 10000
    TEST_OBS = 20000
    SEED = 0
    
    OUTPUT_FOLDER = "testruns"
    WBN = "test"

rlace_optimizer_class = torch.optim.SGD
rlace_scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
SCHED_MIN_LR = PLR * (SCHED_FACTOR**SCHED_NRED)

rlace_optimizer_params_P = {"lr": PLR, 
                            "weight_decay": 1e-4}
rlace_scheduler_params_P = {"mode": "max", 
                            "factor": SCHED_FACTOR, 
                            "patience": SCHED_PATIENCE, 
                            "min_lr": SCHED_MIN_LR, 
                            "verbose": True}
rlace_optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
rlace_epsilon = 0.001 # stop 0.1% from majority acc
rlace_batch_size = 256

# Logging run args
run_args = {
    "rank": RANK,
    "rlace_niter": RLACE_NITER,
    "p_lr": PLR,
    "n_lr_red": SCHED_NRED,
    "nruns": NRUNS,
    "train_obs": TRAIN_OBS,
    "seed": SEED,
    "out_folder": OUTPUT_FOLDER,
    "model_name": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "train_obs": TRAIN_OBS,
    "val_obs": VAL_OBS,
    "test_obs": TEST_OBS
}

RUN_NAME = f"{MODEL_NAME[:4]}_k_{RANK}_n_{RLACE_NITER}_plr_{PLR}"

logging.info(f"Running: {RUN_NAME}")

#%%#################
# Loading Data     #
####################

# Output directory creation
OUTPUT_DIR = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/{MODEL_NAME}/{OUTPUT_FOLDER}/"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    logging.info(f"Created output dir: {OUTPUT_DIR}")
else: 
    logging.info(f"Output dir exists: {OUTPUT_DIR}")

DIAG_RLACE_U_OUTDIR = os.path.join(OUTPUT_DIR, "diag_rlace_u")
if not os.path.exists(DIAG_RLACE_U_OUTDIR):
    os.mkdir(DIAG_RLACE_U_OUTDIR)

# Loading word lists for KL eval
WORD_EMB, SG_EMB, PL_EMB, VERB_PROBS = load_model_eval(MODEL_NAME)

# Load dataset
if MODEL_NAME == "gpt2":
    DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_ar.pkl"
elif MODEL_NAME == "bert-base-uncased":
    DATASET = f"/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/processed/{DATASET_NAME}_{MODEL_NAME}_masked.pkl"
else:
    DATASET = None

with open(DATASET, 'rb') as f:      
    data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])

X = np.array([x for x in data["h"]])
U = np.array([x for x in data["u"]])
y = np.array([yi for yi in data["y"]])
del data

# Set seed
np.random.seed(SEED)

#%%#################
# Wandb Logging    #
####################
if WBN:
    wandb.init(
        project="usagebasedprobing", 
        entity="cguerner",
        name=WBN+f"_{RUN_NAME}",
    )
    wandb.config.update(run_args)
    WB = True
else:
    WB = False


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
    start = time.time()
    
    diag_rlace_u_outfile = os.path.join(DIAG_RLACE_U_OUTDIR, f"{RUN_NAME}.pt")
    
    #dim = X_train.shape[1]

    diag_rlace_output = solve_adv_game(
        X_train, y_train, X_val, y_val, rank=RANK, device=device, 
        out_iters=RLACE_NITER, optimizer_class=rlace_optimizer_class, 
        optimizer_params_P =rlace_optimizer_params_P, 
        optimizer_params_predictor=rlace_optimizer_params_predictor, 
        scheduler_class=rlace_scheduler_class, 
        scheduler_params_P=rlace_scheduler_params_P,
        epsilon=rlace_epsilon,batch_size=rlace_batch_size,
        torch_outfile=diag_rlace_u_outfile, wb=WB, wb_run=i
    )
    end = time.time()
    
    usage_eval = full_usage_eval(
        diag_rlace_output, 
        X_train, U_train, y_train, 
        X_val, U_val, y_val,
        X_test, U_test, y_test, end-start
    )

    kl_eval = compute_kls(
        X_test, diag_rlace_output["P"], diag_rlace_output["I_P"], 
        WORD_EMB, SG_EMB, PL_EMB, VERB_PROBS
    )
    kl_means = kl_eval.loc["mean",:]

    if WB:
        wandb.log({
            f"diag_rlace/test/usage/{i}/diag_acc_P_test": usage_eval["diag_acc_P_test"],
            f"diag_rlace/test/usage/{i}/diag_acc_I_P_test": usage_eval["diag_acc_I_P_test"],
            f"diag_rlace/test/usage/{i}/lm_acc_P_test": usage_eval["lm_acc_P_test"], 
            f"diag_rlace/test/usage/{i}/lm_acc_I_P_test": usage_eval["lm_acc_I_P_test"],
            f"diag_rlace/test/fth_kls/{i}/P_faith_kl_all_split": kl_means["P_faith_kl_all_split"],
            f"diag_rlace/test/fth_kls/{i}/P_faith_kl_all_merged": kl_means["P_faith_kl_all_merged"],
            f"diag_rlace/test/fth_kls/{i}/P_faith_kl_words": kl_means["P_faith_kl_words"],
            f"diag_rlace/test/fth_kls/{i}/P_faith_kl_tgt_split": kl_means["P_faith_kl_tgt_split"],
            f"diag_rlace/test/fth_kls/{i}/P_faith_kl_tgt_merged": kl_means["P_faith_kl_tgt_merged"],
            f"diag_rlace/test/fth_kls/{i}/I_P_faith_kl_all_split": kl_means["I_P_faith_kl_all_split"],
            f"diag_rlace/test/fth_kls/{i}/I_P_faith_kl_all_merged": kl_means["I_P_faith_kl_all_merged"],
            f"diag_rlace/test/fth_kls/{i}/I_P_faith_kl_words": kl_means["I_P_faith_kl_words"],
            f"diag_rlace/test/fth_kls/{i}/I_P_faith_kl_tgt_split": kl_means["I_P_faith_kl_tgt_split"],
            f"diag_rlace/test/fth_kls/{i}/I_P_faith_kl_tgt_merged": kl_means["I_P_faith_kl_tgt_merged"],
            f"diag_rlace/test/er_kls/{i}/P_er_kl_base_proj": kl_means["P_er_kl_base_proj"],
            f"diag_rlace/test/er_kls/{i}/P_er_kl_maj_base": kl_means["P_er_kl_maj_base"],
            f"diag_rlace/test/er_kls/{i}/P_er_kl_maj_proj": kl_means["P_er_kl_maj_proj"],
            f"diag_rlace/test/er_kls/{i}/I_P_er_kl_base_proj": kl_means["I_P_er_kl_base_proj"],
            f"diag_rlace/test/er_kls/{i}/I_P_er_kl_maj_base": kl_means["I_P_er_kl_maj_base"],
            f"diag_rlace/test/er_kls/{i}/I_P_er_kl_maj_proj": kl_means["I_P_er_kl_maj_proj"],
        })
    

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
        diag_rlace_usage_eval = usage_eval,
        diag_rlace_kl_eval = kl_means,
        #functional_rlace = functional_rlace_results,
        #inlp = inlp_results,
        maj_acc_test = get_majority_acc(y_test),
        maj_acc_val = get_majority_acc(y_val),
        maj_acc_train = get_majority_acc(y_train)
    )
    
    #%%
    outfile_path = os.path.join(OUTPUT_DIR, f"run_{RUN_NAME}_{i}_{NRUNS}.pkl")

    with open(outfile_path, 'wb') as f:
        pickle.dump(full_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported {outfile_path}")

    #if WB:
    #    run.finish()


logging.info("Done")

# %%
