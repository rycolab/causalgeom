#%%
import warnings
import logging
import os
import sys
import argparse
import coloredlogs

import pickle
from tqdm import tqdm, trange
from datetime import datetime
import numpy as np
#import wandb

#sys.path.append('..')
sys.path.append('./src/')

from utils.config_args import get_train_probes_config
from algorithms.ProjectionTrainer import ProjectionTrainer

from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# MAIN             #
####################
if __name__ == '__main__':
    cfg = get_train_probes_config()

    logging.info(f"Running: {cfg['run_name']}")

    # Output directory creation
    OUTPUT_DIR = os.path.join(OUT, 
        f"run_output/{cfg['concept']}/{cfg['model_name']}/"
        f"{cfg['out_folder']}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set seed
    np.random.seed(cfg['seed'])

    # Date
    datetimestr = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    
    ####################
    # Wandb Logging    #
    ####################
    #if cfg['wandb_name'] is not None:
    #    logging.info(f"Logging to wandb turned ON, logging to: {cfg['wandb_name']}")
    #    wandb.init(
    #        project="usagebasedprobing", 
    #        entity="cguerner",
    #        name=f"{cfg['out_folder']}_{cfg['wandb_name']}_{cfg['run_name']}_{datetimestr}"
    #    )
    #    wandb.config.update(cfg)
    #    WB = True
    #else:
    #    logging.info(f"Logging to wandb turned OFF")
    #    WB = False

    for i in trange(cfg['nruns']):
        trainer = ProjectionTrainer(
            cfg['model_name'], cfg['concept'], cfg['source'], 
            cfg['train_nsamples'], cfg['val_nsamples'], cfg['test_nsamples']
        )
        run_output = trainer.train_and_format()

        run_output["config"] = cfg
        run_output["i"] = i

        outfile_path = os.path.join(OUTPUT_DIR, 
            f"run_{cfg['run_name']}_{datetimestr}_{i}_{cfg['nruns']}.pkl")

        with open(outfile_path, 'wb') as f:
            pickle.dump(run_output, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Exported {outfile_path}")    

    logging.info("Done")
