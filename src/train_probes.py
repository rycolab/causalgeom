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

#from rlace import solve_adv_game, solve_adv_game_param_free, \
#    init_classifier, get_majority_acc, solve_adv_game_param_free_twoPs
from algorithms.rlace.rlace import solve_adv_game, init_classifier, get_majority_acc, solve_adv_game_param_free

import algorithms.inlp.debias
from classifiers.classifiers import BinaryParamFreeClf
#from classifiers.compute_marginals import compute_concept_marginal, compute_pair_marginals
from utils.cuda_loaders import get_device
from utils.config_args import get_train_probes_config
from evals.kl_eval import load_model_eval, compute_eval_filtered_hs
from evals.usage_eval import full_usage_eval, full_diag_eval
from utils.dataset_loaders import load_processed_data
from data.embed_wordlists.embedder import load_concept_token_lists


from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

def get_data_indices(nobs, cfg):
    idx = np.arange(0, nobs)
    np.random.shuffle(idx)

    if cfg["concept"] == "number":
        train_lastind = cfg['train_obs']
        val_lastind = train_lastind + cfg['val_obs']
        test_lastind = val_lastind + cfg['test_obs']
    elif cfg["concept"] == "gender":
        train_lastind = int(nobs*cfg["train_share"])
        val_lastind = int(nobs*(cfg["train_share"] + cfg["val_share"]))
        test_lastind = nobs
    else:
        raise ValueError("Concept value not supported.")
    return idx[:train_lastind], idx[train_lastind:val_lastind], idx[val_lastind:test_lastind]

def train_probes(X, U, y, facts, foils, 
    cfg, wb, wb_run, diag_rlace_u_outdir, 
    l0_tl, l1_tl, device="cpu"):
    idx_train, idx_val, idx_test = get_data_indices(X.shape[0], cfg)
    
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    U_train, U_val, U_test = U[idx_train], U[idx_val], U[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    facts_train, facts_val, facts_test = facts[idx_train], facts[idx_val], facts[idx_test]
    foils_train, foils_val, foils_test = foils[idx_train], foils[idx_val], foils[idx_test]
    #%%
    #if cfg['pca_dim'] > 0:
    #    logging.info(f"Applying PCA with dimension {cfg['pca_dim']}")
    #    X_pca = PCA(n_components=cfg['pca_dim']) 
    #    X_train_pca = X_pca.fit_transform(X_train)
        #U_pca = PCA(n_components=cfg['pca_dim'])
        #U_train = U_pca.fit_transform(U_train)

    #    X_val_pca = X_pca.transform(X_val)
    #    X_test_pca = X_pca.transform(X_test)
        #U_val = U_pca.transform(U_val)
        #U_test = U_pca.transform(U_test)
    #else:
    #    X_pca = None
    #    X_train_pca = X_train
    #    X_val_pca = X_val
    #    X_test_pca = X_test
    X_pca = None

    #%%
    start = time.time()
    
    diag_rlace_u_outfile = os.path.join(diag_rlace_u_outdir, f"{cfg['run_name']}.pt")
    
    #dim = X_train.shape[1]

    # ESTIMATE MARGINALS VIA COUNTS
    #concept_marginal = compute_concept_marginal(y_train)
    #pair_marginals = compute_pair_marginals(y_train, fact_train, foil_train)

    # ESTIMATE P
    rlace_optimizer_class = torch.optim.SGD
    #rlace_scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    rlace_scheduler_class = torch.optim.lr_scheduler.MultiStepLR

    if cfg["k"] > 0 and cfg["rlace_type"] == "theta":
        rlace_output = solve_adv_game(
            X_train, y_train, X_val, y_val, rank=cfg['k'], device=device, 
            out_iters=cfg['niter'], optimizer_class=rlace_optimizer_class, 
            optimizer_params_P=cfg['rlace_optimizer_params_P'], 
            optimizer_params_predictor=cfg['rlace_optimizer_params_clf'], 
            scheduler_class=rlace_scheduler_class, 
            scheduler_params_P=cfg['rlace_scheduler_params_P'],
            scheduler_params_predictor=cfg['rlace_scheduler_params_clf'],
            batch_size=cfg['batch_size'],
            torch_outfile=diag_rlace_u_outfile, wb=wb, wb_run=wb_run,
            concept=cfg["concept"], model_name=cfg["model_name"],
            X_pca=X_pca
        )
    elif cfg["k"] > 0 and cfg["rlace_type"] == "lm":
        rlace_output = solve_adv_game_param_free(
            X_train, U_train, y_train, X_val, U_val, y_val, 
            version="original", rank=cfg['k'], device=device, 
            out_iters=cfg['niter'], optimizer_class=rlace_optimizer_class, 
            batch_size=cfg['batch_size'],
            wb=wb, wb_run=wb_run,
            concept=cfg["concept"], model_name=cfg["model_name"]
        )
    elif cfg["k"] == 0:
        I = np.eye(X_train.shape[1])
        rlace_output = {
            "P_burn": I,
            "I_P_burn": I
        }
    else: 
        raise ValueError("Incorrect RLACE type")
    
    end = time.time()
    rlace_output["runtime"] = end-start
    
    logging.info("Computing evals")

    diag_eval = full_diag_eval(
        rlace_output, X_train[:50000], y_train[:50000], X_val, y_val, 
        X_test, y_test
    )
    usage_eval = full_usage_eval(
        rlace_output, X_train, U_train, y_train, X_test, U_test, y_test
    )
    # TODO: this nsamples stuff is kinda meh
    #nsamples = 1000
    #l0_hs_wff_test, l1_hs_wff_test = filter_hs_wff(
    #    X_test, facts_test, foils_test, l0_tl, l1_tl, nsamples
    #)
    #new_eval = compute_eval_filtered_hs(
    #    cfg["model_name"], cfg["concept"], rlace_output["P_burn"], 
    #    rlace_output["I_P_burn"], l0_hs_wff_test, l1_hs_wff_test
    #)
    
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
        rlace_type=cfg["rlace_type"],
        output=rlace_output,
        diag_eval=diag_eval,
        usage_eval=usage_eval,
        #new_eval=new_eval,
        maj_acc_test=get_majority_acc(y_test),
        maj_acc_val=get_majority_acc(y_val),
        maj_acc_train=get_majority_acc(y_train),
        X_test=X_test,
        U_test=U_test,
        y_test=y_test,
        foils_test=foils_test,
        facts_test=facts_test,
        #X_pca=X_pca,
    )
    
    return full_results


#%%#################
# MAIN     #
####################
if __name__ == '__main__':
    device = get_device()
    cfg = get_train_probes_config()

    logging.info(f"Running: {cfg['run_name']}")

    # Output directory creation
    OUTPUT_DIR = os.path.join(OUT, 
        f"run_output/{cfg['concept']}/{cfg['model_name']}/"
        f"{cfg['out_folder']}/")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output dir: {OUTPUT_DIR}")
    else: 
        logging.info(f"Output dir exists: {OUTPUT_DIR}")

    DIAG_RLACE_U_OUTDIR = os.path.join(OUTPUT_DIR, "diag_rlace_u")
    if not os.path.exists(DIAG_RLACE_U_OUTDIR):
        os.mkdir(DIAG_RLACE_U_OUTDIR)

    # Loading word lists for KL eval
    #other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(cfg['concept'], cfg['model_name'])

    # Load dataset
    l0_tl, l1_tl = load_concept_token_lists(cfg['concept'], cfg['model_name'])
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
        #X, U, y, cfg, wb, wb_run, diag_rlace_u_outdir, device="cpu"
        run_output = train_probes(X, U, y, facts, foils, cfg, WB, i, 
            DIAG_RLACE_U_OUTDIR, l0_tl, l1_tl, device=device)        

        outfile_path = os.path.join(OUTPUT_DIR, 
            f"run_{cfg['run_name']}_{datetimestr}_{i}_{cfg['nruns']}.pkl")

        with open(outfile_path, 'wb') as f:
            pickle.dump(run_output, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Exported {outfile_path}")    

    logging.info("Done")
