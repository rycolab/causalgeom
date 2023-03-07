import warnings
import logging
import os
import sys
import argparse
import coloredlogs

#%%#################
# Args             #
####################
def get_train_probes_args():
    argparser = argparse.ArgumentParser(description='Train RLACE and evaluate P')
    argparser.add_argument(
        "-out_folder",
        type=str,
        default="test",
        help="Directory for exporting run eval"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=["gpt2", "bert-base-uncased"],
        required=True,
        dest="model_name",
        default="bert-base-uncased",
        help="Model used to extract hidden states & embeddings"
    )
    argparser.add_argument(
        "-k",
        type=int,
        default=1,
        help="Rank of P."
    )
    argparser.add_argument(
        "-niter",
        type=int,
        default=75000,
        help="Number of iterations of RLACE"
    )
    argparser.add_argument(
        "-bs",
        type=int,
        default=256,
        help="Batch size of RLACE"
    )
    argparser.add_argument(
        "-P_lr",
        type=float,
        help="Learning rate for P" 
    )
    argparser.add_argument(
        "-P_n_lr_red",
        type=int,
        default=5,
        help="Number of ReduceLROnPlateau reductions for P" 
    )
    argparser.add_argument(
        "-clf_lr",
        type=float,
        help="Learning rate for clf" 
    )
    argparser.add_argument(
        "-clf_n_lr_red",
        type=int,
        default=5,
        help="Number of ReduceLROnPlateau reductions for CLF" 
    )
    argparser.add_argument(
        "-nruns",
        type=int,
        default=1,
        help="Number of runs of the experiment"
    )
    argparser.add_argument(
        "-train_obs",
        type=int,
        default=200000,
        help="Number of train obs"
    )
    argparser.add_argument(
        "-seed",
        type=int,
        default=0,
        help="Seed for shuffling data"
    )
    argparser.add_argument(
        '-wbn', 
        dest='wandb_name', 
        default="test", 
        type=str, 
        help="Name of wandb run."
    )
    return vars(argparser.parse_args())

#%% Helpers
def get_default_lrs(model_name):
    p_lr = None
    clf_lr = None
    if model_name == "gpt2":
        p_lr = 0.001
        clf_lr = 0.0003
    elif model_name == "bert-base-uncased":
        p_lr = 0.003
        clf_lr = 0.003
    else:
        raise ValueError("Incorrect model name")
    return p_lr, clf_lr


def set_train_probes_defaults(config):
    # Main defaults
    config["dataset_name"] = "linzen"
    
    # Default LRs
    default_P_lr, default_clf_lf = get_default_lrs(
        config["model_name"])
    if not "P_lr" in config.keys():
        config["P_lr"] = default_P_lr
    if not "clf_lr" in config.keys():
        config["clf_lr"] = default_clf_lr

    # P Scheduler
    config["P_sched_factor"] = .5
    config["P_sched_patience"] = 4
    config["P_sched_min_lr"] = (
        config["P_lr"] * (config["P_sched_factor"]**config["P_n_lr_red"])
    )

    # clf Scheduler
    config["clf_sched_factor"] = .5
    config["clf_sched_patience"] = 4
    config["clf_sched_min_lr"] = (
        config["clf_lr"] * (config["clf_sched_factor"]**config["clf_n_lr_red"])
    )

    # Val and test size
    config["val_obs"] = 10000
    config["test_obs"] = 20000

    # Constructing RLACE arg dicts (DON'T SET DEFAULTS HERE)
    config["rlace_optimizer_params_P"] = {
        "lr": config["P_lr"], 
        "weight_decay": 1e-4
    }
    config["rlace_scheduler_params_P"] = {
        "mode": "max", 
        "factor": config["P_sched_factor"], 
        "patience": config["P_sched_patience"], 
        "min_lr": config["P_sched_min_lr"], 
        "verbose": True
    }
    config["rlace_optimizer_params_clf"] = {
        "lr": config["clf_lr"],
        "weight_decay": 1e-4
    }
    config["rlace_scheduler_params_clf"] = {
        "mode": "min", 
        "factor": config["clf_sched_factor"], 
        "patience": config["clf_sched_patience"], 
        "min_lr": config["clf_sched_min_lr"], 
        "verbose": True
    }
    #rlace_epsilon = 0.001 # stop 0.1% from majority acc (I TURNED THIS OFF)
    config["run_name"] = f"{config["model_name"][:4]}_k_{config["k"]}"
    return config

def get_train_probes_config():
    config = get_train_probes_args()
    logging.info(args)

    config = set_train_probes_defaults(config)
    return config
    




"""
MODE = "job" # "debug"


if MODE == "job":
    # rlace args
    RANK = args.k
    RLACE_NITER = args.niter
    BATCH_SIZE = args.bs
    #P_LR = args.p_lr
    #CLF_LR = args.clf_lr
    P_LR, CLF_LR = get_default_lrs(MODEL_NAME)

    # P scheduler
    P_SCHED_NRED = args.P_n_lr_red
    P_SCHED_FACTOR = .5
    P_SCHED_PATIENCE = 4

    # clf scheduler
    CLF_SCHED_NRED = args.clf_n_lr_red
    CLF_SCHED_FACTOR = .5
    CLF_SCHED_PATIENCE = 4

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
    
    # rlace
    RANK = 1
    RLACE_NITER = 1000
    BATCH_SIZE = 256
    #P_LR=0.003
    #CLF_LR = 0.003
    P_LR, CLF_LR = get_default_lrs(MODEL_NAME)

    # P scheduler
    P_SCHED_NRED = 5
    P_SCHED_FACTOR = .5
    P_SCHED_PATIENCE = 4

    # clf scheduler
    CLF_SCHED_NRED = 5
    CLF_SCHED_FACTOR = .5
    CLF_SCHED_PATIENCE = 4

    #data
    NRUNS = 1
    TRAIN_OBS = 60000
    VAL_OBS = 10000
    TEST_OBS = 20000
    SEED = 0
    
    OUTPUT_FOLDER = "testruns"
    WBN = "test"

rlace_optimizer_params_P = {"lr": P_LR, 
                            "weight_decay": 1e-4}
rlace_scheduler_params_P = {"mode": "max", 
                            "factor": SCHED_FACTOR, 
                            "patience": SCHED_PATIENCE, 
                            "min_lr": SCHED_MIN_LR, 
                            "verbose": True}
rlace_optimizer_params_clf = {"lr": CLF_LR,"weight_decay": 1e-4}
#rlace_scheduler_params_predictor = {"mode": "min", 
#                            "factor": SCHED_FACTOR, 
#                            "patience": SCHED_PATIENCE, 
#                            "min_lr": SCHED_MIN_LR, 
#                            "verbose": True}
rlace_epsilon = 0.001 # stop 0.1% from majority acc
rlace_batch_size = BATCH_SIZE

# Logging run args
run_args = {
    "rank": RANK,
    "rlace_niter": RLACE_NITER,
    "batch_size": BATCH_SIZE,
    "p_lr": P_LR,
    "clf_lr": CLF_LR,
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

RUN_NAME = 

"""