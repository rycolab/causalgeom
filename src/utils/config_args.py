import warnings
import logging
import os
import sys
import argparse
import coloredlogs

from utils.lm_loaders import GPT2_LIST, BERT_LIST, SUPPORTED_AR_MODELS
from paths import FR_DATASETS

#%%#################
# Args             #
####################
def get_train_probes_args():
    argparser = argparse.ArgumentParser(description='Train RLACE and evaluate P')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["number", "gender", "food", "ambiance", "service", "noise"],
        default="number",
        help="Concept to erase"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        #required=True,
        dest="model_name",
        default="gpt2-large",
        help="Model used to extract hidden states & embeddings"
    )
    argparser.add_argument(
        "-out_folder",
        type=str,
        default="test",
        help="Directory for exporting run eval"
    )

    # RLACE ARGS
    ##argparser.add_argument(
    ##    "-rlace_type",
    ##    type=str,
    ##    choices=["theta","lm","leace"],
    ##    default="theta",
    ##    help="Which type of RLACE to use"
    ##)
    argparser.add_argument(
        "-k",
        type=int,
        default=1,
        help="Rank of P."
    )
    argparser.add_argument(
        "-niter",
        type=int,
        default=10000,
        help="Number of iterations of RLACE"
    )
    argparser.add_argument(
        "-bs",
        type=int,
        default=256,
        dest="batch_size",
        help="Batch size of RLACE"
    )
    argparser.add_argument(
        "-pca_dim",
        type=int,
        default=0,
        help="Dimension of PCA"
    )
    argparser.add_argument(
        "-P_lr",
        type=float,
        help="Learning rate for P" 
    )
    argparser.add_argument(
        "-P_momentum",
        type=float,
        default=0,
        help="SGD momentum for P" 
    )
    argparser.add_argument(
        "-P_step_size",
        type=int,
        help="StepLR period of learning rate decay for P" 
    )
    argparser.add_argument(
        "-P_milestones",
        default="4,9",
        type=str,
        help="MultiStepLR milestones of learning rate decay for P, has to have format 10,20,30..." 
    )
    argparser.add_argument(
        "-P_gamma",
        type=float,
        default=0.5,
        help="StepLR multiplicative factor of learning rate decay for P" 
    )
    argparser.add_argument(
        "-P_n_lr_red",
        type=int,
        default=5,
        help="Number of ReduceLROnPlateau reductions for P" 
    )
    argparser.add_argument(
        "-P_sched_patience",
        type=int,
        help="Patience parameter of ReduceLROnPlateau for P" 
    )
    argparser.add_argument(
        "-clf_lr",
        type=float,
        help="Learning rate for clf" 
    )
    argparser.add_argument(
        "-clf_momentum",
        type=float,
        default=0,
        help="SGD momentum for clf" 
    )
    argparser.add_argument(
        "-clf_step_size",
        type=int,
        help="StepLR period of learning rate decay for clf" 
    )
    argparser.add_argument(
        "-clf_milestones",
        type=str,
        default="5,10",
        help="MultiStepLR milestones of learning rate decay for clf, has to have format 10,20,30..." 
    )
    argparser.add_argument(
        "-clf_gamma",
        type=float,
        default=0.5,
        help="StepLR multiplicative factor of learning rate decay for clf" 
    )
    argparser.add_argument(
        "-clf_n_lr_red",
        type=int,
        help="Number of ReduceLROnPlateau reductions for clf" 
    )
    argparser.add_argument(
        "-clf_sched_patience",
        type=int,
        help="Patience parameter of ReduceLROnPlateau for clf" 
    )

    # TRAINING ARGS
    argparser.add_argument(
        "-nruns",
        type=int,
        default=1,
        help="Number of runs of the experiment"
    )
    #argparser.add_argument(
    #    "-train_obs",
    #    type=int,
    #    default=200000,
    #    help="Number of train obs"
    #)
    argparser.add_argument(
        "-seed",
        type=int,
        default=0,
        help="Seed for shuffling data"
    )
    argparser.add_argument(
        '-wbn', 
        dest='wandb_name', 
        default=None, 
        type=str, 
        help="Name of wandb run."
    )
    return vars(argparser.parse_args())

#%% Helpers
def get_rlace_model_defaults(model_name):
    if model_name in GPT2_LIST:
        defaults = dict(
            P_lr = 0.001,
            P_sched_patience=10,
            clf_lr = 0.0003,
            clf_n_lr_red = 5,
            clf_sched_patience=10
        )
    elif model_name in BERT_LIST:
        defaults = dict(
            P_lr=0.003,
            P_sched_patience=4,
            clf_lr=0.003,
            clf_n_lr_red=0,
            clf_sched_patience=0,
        )
    else:
        defaults = {}
        raise NotImplementedError(
            f"Model {model_name} has no RLACE params")
    return defaults

def set_train_probes_defaults(config):
    # Default LRs
    #model_defaults = get_rlace_model_defaults(config["model_name"])
    #for key, val in model_defaults.items():
    #    if config[key] is None:
    #        config[key] = val

    # P Scheduler
    #config["P_sched_factor"] = .5
    #config["P_sched_min_lr"] = (
    #    config["P_lr"] * (config["P_sched_factor"]**config["P_n_lr_red"])
    #)

    # clf Scheduler
    #config["clf_sched_factor"] = .5
    #config["clf_sched_min_lr"] = (
    #    config["clf_lr"] * (config["clf_sched_factor"]**config["clf_n_lr_red"])
    #)

    # Train, val and test size
    config["train_obs"] = 60000
    config["val_obs"] = 20000
    config["test_obs"] = 20000

    config["train_share"] = .6
    config["val_share"] = .2
    config["test_share"] = .2

    # Constructing RLACE arg dicts (DON'T SET DEFAULTS HERE)
    #config["rlace_optimizer_params_P"] = {
    #    "lr": config["P_lr"],
    #    "momentum": config["P_momentum"],
    #    "weight_decay": 1e-4
    #}
    #config["rlace_scheduler_params_P"] = {
    #    "mode": "max", 
    #    "factor": config["P_sched_factor"], 
    #    "patience": config["P_sched_patience"], 
    #    "min_lr": config["P_sched_min_lr"], 
    #    "verbose": True
    #}
    #def format_milestones(mstr):
    #    return [int(x) for x in mstr.split(",")]

    #config["rlace_scheduler_params_P"] = {
    #    #"step_size": config["P_step_size"], 
    #    "milestones": format_milestones(config["P_milestones"]), 
    #    "gamma": config["P_gamma"],
    #    "verbose": True
    #}

    #config["rlace_optimizer_params_clf"] = {
    #    "lr": config["clf_lr"],
    #    "momentum": config["clf_momentum"],
    #    "weight_decay": 1e-4
    #}
    #config["rlace_scheduler_params_clf"] = {
    #    "mode": "min", 
    #    "factor": config["clf_sched_factor"], 
    #    "patience": config["clf_sched_patience"], 
    #    "min_lr": config["clf_sched_min_lr"], 
    #    "verbose": True
    #}
    #config["rlace_scheduler_params_clf"] = {
    #    #"step_size": config["clf_step_size"], 
    #    "milestones": format_milestones(config["clf_milestones"]), 
    #    "gamma": config["clf_gamma"],
    #    "verbose": True
    #}
    #rlace_epsilon = 0.001 # stop 0.1% from majority acc (I TURNED THIS OFF)
    #config["run_name"] = f"{config['model_name']}_k{config['k']}_Plr{config['P_lr']}_Pms{config['P_milestones']}_clflr{config['clf_lr']}_clfms{config['clf_milestones']}"
    config["run_name"] = f"leace_{config['concept']}_{config['model_name']}"
    return config

def get_train_probes_config():
    config = get_train_probes_args()
    logging.info(config)

    config = set_train_probes_defaults(config)
    return config
    
