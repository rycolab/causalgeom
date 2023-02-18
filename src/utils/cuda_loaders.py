import warnings
import logging
import os
import sys
import coloredlogs
import torch

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU found, model: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU info: {torch.cuda.get_device_properties(0)}")
    else: 
        device = torch.device("cpu")
        logging.warning("No GPU found")
    return device