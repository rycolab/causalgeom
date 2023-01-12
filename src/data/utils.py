import sys
sys.path.append('./src/')

import numpy as np
import pandas as pd
from datasets import load_dataset

from paths import HF_CACHE

def compute_unique_counts(token_array, count_sort = False):
    token_ids, counts = np.unique(token_array, return_counts = True)
    if count_sort:
        count_sort_ind = np.argsort(counts)
        token_ids = token_ids[count_sort_ind]
        counts = counts[count_sort_ind]
    return token_ids, counts

def merge_token_ids(token_ids_1, vals_1, token_ids_2, vals_2):
    series_1 = pd.Series(vals_1, index = token_ids_1)
    series_2 = pd.Series(vals_2, index = token_ids_2)
    combo = pd.concat([series_1, series_2], axis=1)
    combo.fillna(0, inplace=True)
    combo["total"] = combo[0] + combo[1]
    token_ids = combo.index.to_numpy()
    vals = combo.total.to_numpy()
    return token_ids, vals

def get_dataset(name):
    if name == "bookcorpus":
        ds = load_dataset(
            "bookcorpus", 
            cache_dir=HF_CACHE
        )
    elif name == "wikipedia":
        ds = load_dataset(
            "wikipedia", 
            "20220301.en", 
            cache_dir=HF_CACHE
        )
    else:
        raise Exception("Dataset value not supported")
    return ds