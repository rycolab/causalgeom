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
import wandb

#from rlace import solve_adv_game, solve_adv_game_param_free, \
#    init_classifier, get_majority_acc, solve_adv_game_param_free_twoPs
from algorithms.rlace.rlace import solve_adv_game, init_classifier, get_majority_acc

import algorithms.inlp.debias
from classifiers.classifiers import BinaryParamFreeClf
from utils.cuda_loaders import get_device
from utils.config_args import get_train_probes_config
from evals.kl_eval import compute_kls, load_model_eval
from evals.usage_eval import full_usage_eval, full_diag_eval
from data.dataset_loaders import load_processed_data

from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

device = get_device()
cfg = get_train_probes_config()

rlace_optimizer_class = torch.optim.SGD
#rlace_scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
rlace_scheduler_class = torch.optim.lr_scheduler.MultiStepLR

logging.info(f"Running: {cfg['run_name']}")

#%%#################
# Loading Data     #
####################

# Output directory creation
OUTPUT_DIR = os.path.join(OUT, 
    f"run_output/{cfg['dataset_name']}/{cfg['model_name']}/"
    f"{cfg['out_folder']}/")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    logging.info(f"Created output dir: {OUTPUT_DIR}")
else: 
    logging.info(f"Output dir exists: {OUTPUT_DIR}")

DIAG_RLACE_U_OUTDIR = os.path.join(OUTPUT_DIR, "diag_rlace_u")
if not os.path.exists(DIAG_RLACE_U_OUTDIR):
    os.mkdir(DIAG_RLACE_U_OUTDIR)

# Loading word lists for KL eval
WORD_EMB, SG_EMB, PL_EMB, VERB_PROBS, SG_PL_PROB = load_model_eval(
    cfg['dataset_name'], cfg['model_name'])

# Load dataset
X, U, y = load_processed_data(dataset_name, model_name)

# Set seed
np.random.seed(cfg['seed'])

#%%#################
# Wandb Logging    #
####################
datetimestr = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
if cfg['wandb_name']:
    wandb.init(
        project="usagebasedprobing", 
        entity="cguerner",
        name=f"{cfg['out_folder']}_{cfg['wandb_name']}_{cfg['run_name']}_{datetimestr}"
    )
    wandb.config.update(cfg)
    WB = True
else:
    WB = False


#%%
for i in trange(cfg['nruns']):
    
    #%%
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)

    train_lastind = cfg['train_obs']
    val_lastind = train_lastind + cfg['val_obs']
    test_lastind = val_lastind + cfg['test_obs']
    X_train, X_val, X_test = X[idx[:train_lastind]], X[idx[train_lastind:val_lastind]], X[idx[val_lastind:test_lastind]]
    U_train, U_val, U_test = U[idx[:train_lastind]], U[idx[train_lastind:val_lastind]], U[idx[val_lastind:test_lastind]]
    y_train, y_val, y_test = y[idx[:train_lastind]], y[idx[train_lastind:val_lastind]], y[idx[val_lastind:test_lastind]]

    #%%
    start = time.time()
    
    diag_rlace_u_outfile = os.path.join(DIAG_RLACE_U_OUTDIR, f"{cfg['run_name']}.pt")

    diag_rlace_output = solve_adv_game(
        X_train, y_train, X_val, y_val, rank=cfg['k'], device=device, 
        out_iters=cfg['niter'], optimizer_class=rlace_optimizer_class, 
        optimizer_params_P=cfg['rlace_optimizer_params_P'], 
        optimizer_params_predictor=cfg['rlace_optimizer_params_clf'], 
        scheduler_class=rlace_scheduler_class, 
        scheduler_params_P=cfg['rlace_scheduler_params_P'],
        scheduler_params_predictor=cfg['rlace_scheduler_params_clf'],
        batch_size=cfg['batch_size'],
        torch_outfile=diag_rlace_u_outfile, wb=WB, wb_run=i,
        dataset_name=cfg["dataset_name"], model_name=cfg["model_name"]
    )
    end = time.time()
    diag_rlace_output["runtime"] = end-start
    
    logging.info("Computing evals")

    diag_eval = full_diag_eval(
        diag_rlace_output, X_train[:50000], y_train[:50000], 
        X_val, y_val, X_test, y_test
    )
    usage_eval = full_usage_eval(
        diag_rlace_output, X_train, U_train, y_train, X_test, U_test, y_test
    )

    kl_eval = compute_kls(
        X_test, diag_rlace_output["P"], diag_rlace_output["I_P"], 
        WORD_EMB, SG_EMB, PL_EMB, VERB_PROBS, SG_PL_PROB
    )
    kl_means = kl_eval.loc["mean",:]

    burn_kl_eval = compute_kls(
        X_test, diag_rlace_output["P_burn"], diag_rlace_output["I_P_burn"], 
        WORD_EMB, SG_EMB, PL_EMB, VERB_PROBS, SG_PL_PROB
    )
    burn_kl_means = burn_kl_eval.loc["mean",:]

    if WB:
        wandb.log({
            f"diag_rlace/test/P_burn/diag/{i}/diag_acc_test": diag_eval["diag_acc_P_burn_test"],
            f"diag_rlace/test/I_P_burn/diag/{i}/diag_acc_test": diag_eval["diag_acc_I_P_burn_test"],
            f"diag_rlace/test/P_burn/usage/{i}/lm_acc_test": usage_eval["lm_acc_P_burn_test"], 
            f"diag_rlace/test/I_P_burn/usage/{i}/lm_acc_test": usage_eval["lm_acc_I_P_burn_test"],
            

            f"diag_rlace/test/base/er_mis/{i}/overall_mi": kl_means["base_overall_mi"],
            f"diag_rlace/test/base/er_mis/{i}/pairwise_mi": kl_means["base_pairwise_mi"],

            f"diag_rlace/test/P_burn/fth_kls/{i}/faith_kl_all_split": burn_kl_means["P_faith_kl_all_split"],
            f"diag_rlace/test/P_burn/fth_kls/{i}/faith_kl_all_merged": burn_kl_means["P_faith_kl_all_merged"],
            f"diag_rlace/test/P_burn/fth_kls/{i}/faith_kl_words": burn_kl_means["P_faith_kl_words"],
            f"diag_rlace/test/P_burn/fth_kls/{i}/faith_kl_tgt_split": burn_kl_means["P_faith_kl_tgt_split"],
            f"diag_rlace/test/P_burn/fth_kls/{i}/faith_kl_tgt_merged": burn_kl_means["P_faith_kl_tgt_merged"],

            f"diag_rlace/test/P_burn/er_mis/{i}/overall_mi": burn_kl_means["P_overall_mi"],
            f"diag_rlace/test/P_burn/er_mis/{i}/pairwise_mi": burn_kl_means["P_pairwise_mi"],

            f"diag_rlace/test/I_P_burn/fth_kls/{i}/faith_kl_all_split": burn_kl_means["I_P_faith_kl_all_split"],
            f"diag_rlace/test/I_P_burn/fth_kls/{i}/faith_kl_all_merged": burn_kl_means["I_P_faith_kl_all_merged"],
            f"diag_rlace/test/I_P_burn/fth_kls/{i}/faith_kl_words": burn_kl_means["I_P_faith_kl_words"],
            f"diag_rlace/test/I_P_burn/fth_kls/{i}/faith_kl_tgt_split": burn_kl_means["I_P_faith_kl_tgt_split"],
            f"diag_rlace/test/I_P_burn/fth_kls/{i}/faith_kl_tgt_merged": burn_kl_means["I_P_faith_kl_tgt_merged"],

            f"diag_rlace/test/I_P_burn/er_mis/{i}/overall_mi": burn_kl_means["I_P_overall_mi"],
            f"diag_rlace/test/I_P_burn/er_mis/{i}/pairwise_mi": burn_kl_means["I_P_pairwise_mi"],
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
        run=i,
        config=cfg,
        output=diag_rlace_output,
        diag_eval=diag_eval,
        usage_eval=usage_eval,
        kl_eval=kl_means,
        burn_kl_mean=burn_kl_eval.loc["mean",:],
        burn_kl_std=burn_kl_eval.loc["std",:],
        maj_acc_test=get_majority_acc(y_test),
        maj_acc_val=get_majority_acc(y_val),
        maj_acc_train=get_majority_acc(y_train)
    )
    
    #%%
    outfile_path = os.path.join(OUTPUT_DIR, f"run_{cfg['run_name']}_{datetimestr}_{i}_{cfg['nruns']}.pkl")

    with open(outfile_path, 'wb') as f:
        pickle.dump(full_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Exported {outfile_path}")

    #if WB:
    #    run.finish()


logging.info("Done")

# %%
