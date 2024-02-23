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
from datetime import datetime

import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import wandb
from concept_erasure import LeaceEraser

#from rlace import solve_adv_game, solve_adv_game_param_free, \
#    init_classifier, get_majority_acc, solve_adv_game_param_free_twoPs
from algorithms.rlace.rlace import solve_adv_game, init_classifier, get_majority_acc, solve_adv_game_param_free

import algorithms.inlp.debias
#from classifiers.classifiers import BinaryParamFreeClf
#from classifiers.compute_marginals import compute_concept_marginal, compute_pair_marginals
from utils.cuda_loaders import get_device
from utils.config_args import get_train_probes_config
#from evals.kl_eval import load_model_eval, compute_eval_filtered_hs
#from evals.usage_eval import full_usage_eval, full_diag_eval
from utils.dataset_loaders import load_processed_data
from data.embed_wordlists.embedder import load_concept_token_lists


from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


def get_data_indices(nobs, concept, train_obs, val_obs, test_obs,
        train_share, val_share):
    idx = np.arange(0, nobs)
    np.random.shuffle(idx)

    if concept == "number":
        train_lastind = train_obs
        val_lastind = train_lastind + val_obs
        test_lastind = val_lastind + test_obs
    elif concept == "gender":
        train_lastind = int(nobs*train_share)
        val_lastind = int(nobs*(cfg["train_share"] + val_share))
        test_lastind = nobs
    else:
        raise ValueError("Concept value not supported.")
    return idx[:train_lastind], idx[train_lastind:val_lastind], idx[val_lastind:test_lastind]

def compute_leace_affine(X_train, y_train):
    X_torch = torch.from_numpy(X_train)
    y_torch = torch.from_numpy(y_train)

    eraser = LeaceEraser.fit(X_torch, y_torch)
    #P = (eraser.proj_left @ eraser.proj_right).numpy().T
    #I_P = np.eye(X_train.shape[1]) - P
    #bias = eraser.bias.numpy()

    P = (eraser.proj_right.mH @ eraser.proj_left.mH).numpy()
    I_P = np.eye(X_train.shape[1]) - P
    bias = eraser.bias.numpy()
    return P, I_P, bias


def train_probes(X, U, y, facts, foils,
    idx_train, idx_val, idx_test):  
    #wb, wb_run, l0_tl, l1_tl, 
    #device="cpu"):

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    U_train, U_val, U_test = U[idx_train], U[idx_val], U[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    facts_train, facts_val, facts_test = facts[idx_train], facts[idx_val], facts[idx_test]
    foils_train, foils_val, foils_test = foils[idx_train], foils[idx_val], foils[idx_test]

    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_val shape: {y_val.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    #%%
    start = time.time()

    # RLACE
    #rlace_optimizer_class = torch.optim.SGD
    #rlace_scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    #rlace_scheduler_class = torch.optim.lr_scheduler.MultiStepLR

    #if cfg["k"] > 0 and cfg["rlace_type"] == "theta":
    #    rlace_output = solve_adv_game(
    #        X_train, y_train, X_val, y_val, rank=cfg['k'], device=device, 
    #        out_iters=cfg['niter'], optimizer_class=rlace_optimizer_class, 
    #        optimizer_params_P=cfg['rlace_optimizer_params_P'], 
    #        optimizer_params_predictor=cfg['rlace_optimizer_params_clf'], 
    #        scheduler_class=rlace_scheduler_class, 
    #        scheduler_params_P=cfg['rlace_scheduler_params_P'],
    #        scheduler_params_predictor=cfg['rlace_scheduler_params_clf'],
    #        batch_size=cfg['batch_size'],
    #        torch_outfile=diag_rlace_u_outfile, wb=wb, wb_run=wb_run,
    #        concept=cfg["concept"], model_name=cfg["model_name"],
    #        X_pca=X_pca
    #    )
    #elif cfg["k"] > 0 and cfg["rlace_type"] == "lm":
    #    rlace_output = solve_adv_game_param_free(
    #        X_train, U_train, y_train, X_val, U_val, y_val, 
    #        version="original", rank=cfg['k'], device=device, 
    #        out_iters=cfg['niter'], optimizer_class=rlace_optimizer_class, 
    #        batch_size=cfg['batch_size'],
    #        wb=wb, wb_run=wb_run,
    #        concept=cfg["concept"], model_name=cfg["model_name"]
    #    )
    #elif cfg["k"] == 0 and cfg["rlace_type"] == "theta":
    #    I = np.eye(X_train.shape[1])
    #    rlace_output = {
    #        "P_burn": I,
    #        "I_P_burn": I
    #    }
    #elif cfg["rlace_type"] == "leace":
    
    # LEACE 
    P, I_P, bias = compute_leace_affine(X_train, y_train)
    output = {
        "bias": bias,
        "P": P,
        "I_P": I_P
    }
    #else: 
    #    raise ValueError("Incorrect RLACE type")
    
    end = time.time()
    #rlace_output["runtime"] = end-start
    
    #logging.info("Computing evals")

    #diag_eval = full_diag_eval(
    #    rlace_output, X_train[:50000], y_train[:50000], X_val, y_val, 
    #    X_test, y_test
    #)
    #usage_eval = full_usage_eval(
    #    rlace_output, X_train, U_train, y_train, X_test, U_test, y_test
    #)
    
    #%%
    full_results = dict(
        run=i,
        #config=cfg,
        #rlace_type=cfg["rlace_type"],
        output=output,
        #diag_eval=diag_eval,
        #usage_eval=usage_eval,
        #new_eval=new_eval,
        maj_acc_test=get_majority_acc(y_test),
        maj_acc_val=get_majority_acc(y_val),
        maj_acc_train=get_majority_acc(y_train),
        X_val=X_val,
        U_val=U_val,
        y_val=y_val,
        foils_val=foils_val,
        facts_val=facts_val,
        nobs_train = X_train.shape[0],
        nobs_test = X_test.shape[0],
        X_test=X_test,
        U_test=U_test,
        y_test=y_test,
        foils_test=foils_test,
        facts_test=facts_test,
    )
    
    return full_results


#%%#################
# MAIN             #
####################
if __name__ == '__main__':
    device = get_device()
    cfg = get_train_probes_config()

    logging.info(f"Running: {cfg['run_name']}")

    # Output directory creation
    OUTPUT_DIR = os.path.join(OUT, 
        f"run_output/{cfg['concept']}/{cfg['model_name']}/"
        f"{cfg['out_folder']}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    #l0_tl, l1_tl = load_concept_token_lists(cfg['concept'], cfg['model_name'])
    X, U, y, facts, foils = load_processed_data(cfg['concept'], cfg['model_name'])

    # Set seed
    np.random.seed(cfg['seed'])

    #%%#################
    # Wandb Logging    #
    ####################
    datetimestr = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    if cfg['wandb_name'] is not None:
        logging.info(f"Logging to wandb turned ON, logging to: {cfg['wandb_name']}")
        wandb.init(
            project="usagebasedprobing", 
            entity="cguerner",
            name=f"{cfg['out_folder']}_{cfg['wandb_name']}_{cfg['run_name']}_{datetimestr}"
        )
        wandb.config.update(cfg)
        WB = True
    else:
        logging.info(f"Logging to wandb turned OFF")
        WB = False

    for i in trange(cfg['nruns']):

        idx_train, idx_val, idx_test = get_data_indices(
            X.shape[0], cfg["concept"], 
            cfg['train_obs'], cfg['val_obs'], cfg['test_obs'],
            cfg["train_share"], cfg["val_share"]
        )
        run_output = train_probes(
            X, U, y, facts, foils, 
            idx_train, idx_val, idx_test) 
            #WB, i, 
            #l0_tl, l1_tl, device=device)        
        run_output["config"] = cfg

        outfile_path = os.path.join(OUTPUT_DIR, 
            f"run_{cfg['run_name']}_{datetimestr}_{i}_{cfg['nruns']}.pkl")

        with open(outfile_path, 'wb') as f:
            pickle.dump(run_output, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Exported {outfile_path}")    

    logging.info("Done")
