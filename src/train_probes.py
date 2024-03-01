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
from algorithms.rlace.rlace import get_majority_acc #solve_adv_game, init_classifier, solve_adv_game_param_free

#import algorithms.inlp.debias
#from classifiers.classifiers import BinaryParamFreeClf
#from classifiers.compute_marginals import compute_concept_marginal, compute_pair_marginals
from utils.cuda_loaders import get_device
from utils.config_args import get_train_probes_config
#from evals.kl_eval import load_model_eval, compute_eval_filtered_hs
#from evals.usage_eval import full_usage_eval, full_diag_eval
from utils.dataset_loaders import load_processed_data

from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#TODO: need to make a Trainer class to simplify this garbage
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

def create_run_datasets(X, U, y, facts, foils, cxt_toks, idx_train, idx_dev, idx_test):
    X_train, X_dev, X_test = X[idx_train], X[idx_dev], X[idx_test]
    U_train, U_dev, U_test = U[idx_train], U[idx_dev], U[idx_test]
    y_train, y_dev, y_test = y[idx_train], y[idx_dev], y[idx_test]
    facts_train, facts_dev, facts_test = facts[idx_train], facts[idx_dev], facts[idx_test]
    foils_train, foils_dev, foils_test = foils[idx_train], foils[idx_dev], foils[idx_test]
    cxt_toks_train, cxt_toks_dev, cxt_toks_test = cxt_toks[idx_train], cxt_toks[idx_dev], cxt_toks[idx_test]

    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_dev shape: {y_dev.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    
    return {
        "X_train": X_train, 
        "U_train": U_train,
        "y_train": y_train,
        "facts_train": facts_train,
        "foils_train": foils_train,
        "cxt_toks_train": cxt_toks_train,
        "X_dev": X_dev, 
        "U_dev": U_dev,
        "y_dev": y_dev,
        "facts_dev": facts_dev,
        "foils_dev": foils_dev,
        "cxt_toks_dev": cxt_toks_dev,
        "X_test": X_test, 
        "U_test": U_test,
        "y_test": y_test,
        "facts_test": facts_test,
        "foils_test": foils_test,
        "cxt_toks_test": cxt_toks_test,
    }

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


def train_probes(run_data):  

    X_train, X_dev, X_test = run_data["X_train"], run_data["X_dev"], run_data["X_test"]
    U_train, U_dev, U_test = run_data["U_train"], run_data["U_dev"], run_data["U_test"]
    y_train, y_dev, y_test = run_data["y_train"], run_data["y_dev"], run_data["y_test"]
    facts_train, facts_dev, facts_test = run_data["facts_train"], run_data["facts_dev"], run_data["facts_test"]
    foils_train, foils_dev, foils_test = run_data["foils_train"], run_data["foils_dev"], run_data["foils_test"]
    cxt_toks_train, cxt_toks_dev, cxt_toks_test = run_data["cxt_toks_train"], run_data["cxt_toks_dev"], run_data["cxt_toks_test"]
    
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
        maj_acc_val=get_majority_acc(y_dev),
        maj_acc_train=get_majority_acc(y_train),
        X_val=X_dev,
        U_val=U_dev,
        y_val=y_dev,
        foils_val=foils_dev,
        facts_val=facts_dev,
        cxt_toks_val = cxt_toks_dev,
        nobs_train = X_train.shape[0],
        nobs_test = X_test.shape[0],
        X_test=X_test,
        U_test=U_test,
        y_test=y_test,
        foils_test=foils_test,
        facts_test=facts_test,
        cxt_toks_test = cxt_toks_test,
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
    #TODO: fix this, probably by reshuffling the CEBaB data (though this has implications
    # for what samples are included esp. in dev, test)
    if cfg['concept'] in ["number", "gender"]:
        X, U, y, facts, foils, cxt_toks = load_processed_data(cfg['concept'], cfg['model_name'])
    elif cfg['concept'] in ["food", "ambiance", "service", "noise"]:
        run_data = load_processed_data(cfg['concept'], cfg['model_name'])
    else:
        raise NotImplementedError(f"Concept {cfg['concept']} not supported")

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
        if cfg['concept'] in ["number", "gender"]:    
            idx_train, idx_val, idx_test = get_data_indices(
                X.shape[0], cfg["concept"], 
                cfg['train_obs'], cfg['val_obs'], cfg['test_obs'],
                cfg["train_share"], cfg["val_share"]
            )
            run_data = create_run_datasets(
                X, U, y, facts, foils, cxt_toks, idx_train, idx_val, idx_test
            )
        elif cfg['concept'] in ["food", "ambiance", "service", "noise"]:
            #TODO: probably need to implement some sort of shuffling here 
            None
        else:
            raise NotImplementedError()

        run_output = train_probes(run_data)
        run_output["config"] = cfg

        outfile_path = os.path.join(OUTPUT_DIR, 
            f"run_{cfg['run_name']}_{datetimestr}_{i}_{cfg['nruns']}.pkl")

        with open(outfile_path, 'wb') as f:
            pickle.dump(run_output, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Exported {outfile_path}")    

    logging.info("Done")
