#%%
import os
import sys
import warnings
import logging
import coloredlogs

#sys.path.append('..')
sys.path.append('./src/')

import numpy as np
import random

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%##################
# Dataset Filtering #
#####################
def filter_hs_w_ys(X, facts, foils, y, value):
    idx = np.nonzero(y==value)
    sub_hs, sub_facts, sub_foils = X[idx], facts[idx], foils[idx]
    sub_hs_wff = [x for x in zip(sub_hs, sub_facts, sub_foils)]
    return sub_hs_wff

def sample_filtered_hs(l0_hs, l1_hs, nsamples):
    random.shuffle(l0_hs)
    random.shuffle(l1_hs)
    ratio = len(l1_hs)/len(l0_hs)
    if ratio > 1:
        l0_hs = l0_hs[:nsamples]
        l1_hs = l1_hs[:int((nsamples*ratio))]
    else:
        ratio = len(l0_hs) / len(l1_hs)
        l0_hs = l0_hs[:int((nsamples*ratio))]
        l1_hs = l1_hs[:nsamples]
    return l0_hs, l1_hs 