#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import pickle
import random 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
agg_pairs = [
    ("gpt2-large", "number"),
    #("bert-base-uncased", "number"),
    ("gpt2-base-french", "gender"),
    #("camembert-base", "gender"),
]
#nucfix = True

#if nucfix:
#    rootresdir = os.path.join(RESULTS, "finaleval_bigsamples_nucfix")
#    outfilepath = os.path.join(RESULTS, "finaleval_bigsamples_nucfix_res.csv")
#else:
rootresdir = os.path.join(RESULTS, "leace_eval_final_v2")
#outfilepath = os.path.join(RESULTS, "leace_res.csv")

all_subdirs = []
for model_name, concept in agg_pairs:
    olddir = os.path.join(rootresdir, f"{concept}/{model_name}")
    if os.path.exists(olddir):
        all_subdirs.append(olddir)
    newdir = os.path.join(rootresdir, f"new_{concept}/{model_name}")
    if os.path.exists(newdir):
        all_subdirs.append(newdir)

logging.info(f"Directories being aggregated: {all_subdirs}")

#%%
#subdir = all_subdirs[0]
resrecords = []
for subdir in all_subdirs:
    resfilenames = os.listdir(subdir)
    for resfilename in resfilenames:
        if resfilename.endswith(".pkl"):
            with open(os.path.join(subdir, resfilename), "rb") as f:
                res = pickle.load(f)
            resrecords.append(res)

#%%
df = pd.DataFrame.from_records(resrecords)

#%%
#from final_eval import prep_data

#PC_gpt2large,_,_,_ = prep_data("gpt2-large", False)
# [0.27066386, 0.72933614]
#%%
#PC_gpt2large_nuc,_,_,_ = prep_data("gpt2-large", True)
#%%
#PC_gpt2fr,_,_,_ = prep_data("gpt2-base-french", False)
#%%
#PC_gpt2fr_nuc,_,_,_ = prep_data("gpt2-base-french", True)

#%%
def compute_ent_pxc(model_name, nucleus, l0_ent_pxc, l1_ent_pxc):
    if model_name == "gpt2-large" and nucleus:
        ent_pxc = (PC_gpt2large_nuc * np.array([l0_ent_pxc, l1_ent_pxc])).sum()
    elif model_name == "gpt2-large" and not nucleus:
        ent_pxc = (PC_gpt2large * np.array([l0_ent_pxc, l1_ent_pxc])).sum()
    elif model_name == "gpt2-base-french" and nucleus:
        ent_pxc = (PC_gpt2fr_nuc * np.array([l0_ent_pxc, l1_ent_pxc])).sum()
    elif model_name == "gpt2-base-french" and not nucleus:
        ent_pxc = (PC_gpt2fr * np.array([l0_ent_pxc, l1_ent_pxc])).sum()
    else:
        raise ValueError("Incorrect inputs")
    return ent_pxc

#df["ent_pxc"] = df.apply(
#    lambda x: compute_ent_pxc(
#        x.model_name, x.nucleus, x.l0_ent_pxc, x.l1_ent_pxc
#    )
#)    

#%%
# additional metrics
df["reconstructed"] = df["mi_c_hbot"] + df["mi_c_hpar"]
df["encapsulation"] = df["mi_c_h"] - df["mi_c_hpar"]
df["mi_x_h_c"] = df["ent_pxc"] - df["stab_ent_xhc"]
#percentages
df["perc_mi_c_hbot"] = df["mi_c_hbot"] / df["mi_c_h"]
df["perc_mi_c_hpar"] = df["mi_c_hpar"] / df["mi_c_h"]
df["perc_encapsulation"] = df["encapsulation"] / df["mi_c_h"]
df["perc_reconstructed"] = df["reconstructed"] / df["mi_c_h"]

#%%


#df.to_csv(outfilepath)
#logging.info(f"Exported agg output to: {outfilepath}")
# %%
df["sampling_method"] = df["nucleus"].apply(lambda x: np.where(x, "Nucleus", "Ancestral"))
table_df = df[['model_name', 'concept', 'sampling_method', 
    'mi_c_h', 'mi_c_hbot', 'mi_c_hpar', 
    'reconstructed', 'encapsulation',
    'perc_mi_c_hbot', 'perc_mi_c_hpar',
    'perc_encapsulation', 'perc_reconstructed',
    'cont_mi', 'stab_mi', 'mi_x_h_c', 'ent_pxc',
]]
# %%
mi_renames = {
    "model_name": "Model",
    "concept": "Concept",
    "sampling_method": "Sampling Method",
    'cont_mi': "Contaiment",
    'stab_mi': "Stability",
    'mi_x_h_c': "Baseline", 
    'ent_pxc': "H(X|C)",
    'mi_c_hbot': "Erasure",
    'mi_c_hpar': "Subspace Info",
    'mi_c_h': "Total Info",
    'reconstructed': "Reconstructed MI",
    'encapsulation': "Encapsulation",
    'perc_mi_c_hbot': "Erased %",
    'perc_mi_c_hpar': "Subspace Info %",
    'perc_encapsulation': "Encapsulated %",
    'perc_reconstructed': "Reconstructed %"
}

table_df.sort_values(by = ["model_name", "concept", "sampling_method"], inplace=True)
table_df.columns = [mi_renames[x] for x in table_df.columns]
table_df_grouped = table_df.groupby(["Model", "Concept", "Sampling Method"])
table_df_grouped.mean().reset_index().to_csv(os.path.join(RESULTS, "leace_mis_mean_v2.csv"), index=False)
table_df_grouped.std().reset_index().to_csv(os.path.join(RESULTS, "leace_mis_std_v2.csv"), index=False)
# %%
logging.info("Done")
# %%
