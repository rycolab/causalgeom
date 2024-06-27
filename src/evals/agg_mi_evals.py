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

sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

# %% MI RES

def get_res_file_paths(res_dir, eval_run_name):
    resfolder = os.path.join(RESULTS, f"{res_dir}/{eval_run_name}")
    resfiles = os.listdir(resfolder)
    resfilepaths = [os.path.join(resfolder, x) for x in resfiles]
    return resfilepaths

mifilepaths_1 = get_res_file_paths("mis", "june2")
mifilepaths_2 = get_res_file_paths("mis", "june9_llamanumbers")

mifilepaths = mifilepaths_1 + mifilepaths_2

#%%
res_records = []
for mifile in mifilepaths:
    with open(mifile, 'rb') as f:      
        mires = pickle.load(f)
    res_records.append(mires)

df = pd.DataFrame(res_records)
#%%
# additional metrics
#df["encapsulation"] = df["MIz_c_h"] - df["MIqpar_c_hpar"]
#df["mi_x_h_c"] = df["ent_pxc"] - df["stab_ent_xhc"]
#percentages
#df["perc_mi_c_hbot"] = df["MIqbot_c_hbot"] / df["MIz_c_h"]
#df["perc_mi_c_hpar"] = df["MIqpar_c_hpar"] / df["MIz_c_h"]
#df["perc_encapsulation"] = df["encapsulation"] / df["MIz_c_h"]
#df["perc_reconstructed"] = df["reconstructed"] / df["MIz_c_h"]

#%%
df.drop(
    df[(df["model_name"]=="llama2") & 
        (df["concept"]=="number") &
        (df["eval_name"]=="june2")].index,
    axis=0, inplace=True
)
df.groupby(["concept", "model_name", "proj_source", "eval_source", "eval_name"]).count().to_csv(
    os.path.join(RESULTS, "counts.csv")
)


#%%
df["reconstructed"] = df["MIqbot_c_hbot"] + df["MIqpar_c_hpar"]
df["new_ratio_erasure"] = 1 - (df["MIqbot_c_hbot"] / df["MIz_c_h"])
df["new_ratio_encapsulation"] = df["MIqpar_c_hpar"] / df["MIz_c_h"]
df["new_ratio_reconstructed"] = df["reconstructed"] / df["MIz_c_h"]
df["new_ratio_containment"] = 1 - (df["MIqpar_x_hpar_mid_c"]/df["MIz_x_h_mid_c"])
df["new_ratio_stability"] = df["MIqbot_x_hbot_mid_c"]/df["MIz_x_h_mid_c"]

#%%
proj_source_renames = {
    "natural_all": "CEBaB",
    "gen_ancestral_all": "Gen (Ancestral)",
    "gen_nucleus_all": "Gen (Nucleus)",
}
df["proj_source"] = df["proj_source"].apply(lambda x: proj_source_renames[x])

eval_source_renames = {
    "train_all": "Train (All)",
    "train_concept": "Train (Concept)",
    "test_all": "Test (All)",
    "test_concept": "Test (Concept)",
}
df["eval_source"] = df["eval_source"].apply(lambda x: eval_source_renames[x])

#df.to_csv(outfilepath)
#logging.info(f"Exported agg output to: {outfilepath}")
# %%
#df["sampling_method"] = df["nucleus"].apply(lambda x: np.where(x, "Nucleus", "Ancestral"))
table_df = df[['model_name', 'concept', 'proj_source', 'eval_source', 
    #'mi_c_h', 'mi_c_hbot', 'mi_c_hpar', 
    #'reconstructed', 'encapsulation',
    #'perc_mi_c_hbot', 'perc_mi_c_hpar',
    #'perc_encapsulation', 'perc_reconstructed',
    #'cont_mi', 'stab_mi',  'ent_pxc',
    'MIz_c_h',
    "new_ratio_erasure",
    "new_ratio_encapsulation",
    "new_ratio_reconstructed",
    'MIz_x_h_mid_c',
    "new_ratio_containment",
    "new_ratio_stability",
]]

#%%
#containtest = df[
#    ['model_name', 'concept', 'sampling_method', 
#    "cont_l0_ent_qxhcs", "cont_l1_ent_qxhcs", "cont_ent_qxcs", 
#    "l0_ent_pxc", "l1_ent_pxc", "ent_pxc", "cont_l0_mi", "cont_l1_mi",
#    "cont_mi", 
#    "stab_ent_xhc_l0", "stab_ent_xhc_l1", "stab_ent_xhc", 
#    'ent_pxc', 'mi_x_h_c']
#]
#containtestmean = containtest.groupby(['model_name', 'concept', 'sampling_method']).mean()

#containtestmean.reset_index().to_csv(os.path.join(RESULTS, "debug_containment.csv"), index=False)

# %%
mi_renames = {
    "model_name": "Model",
    "concept": "Concept",
    "proj_source": "Train Source",
    "eval_source": "Test Source",
    'cont_mi': "Contaiment",
    'stab_mi': "Stability",
    'ent_pxc': "H(X|C)",
    'mi_c_hbot': "Erasure",
    'mi_c_hpar': "Subspace Info",
    'reconstructed': "Reconstructed MI",
    'encapsulation': "Encapsulation",
    'perc_mi_c_hbot': "Erased %",
    'perc_mi_c_hpar': "Subspace Info %",
    'perc_encapsulation': "Encapsulated %",
    'perc_reconstructed': "Reconstructed %",
    'MIz_c_h': "I(C;H)",
    'MIz_x_h_mid_c': "I(X;H|C)", 
    "new_ratio_erasure": "Erasure Ratio",
    "new_ratio_encapsulation": "Encapsulation Ratio",
    "new_ratio_reconstructed": "Reconstructed Ratio",
    "new_ratio_containment": "Containment Ratio",
    "new_ratio_stability": "Stability Ratio",
}

table_df.sort_values(by = ["concept", "model_name", "proj_source", "eval_source"], inplace=True)
table_df.columns = [mi_renames[x] for x in table_df.columns]
table_df_grouped = table_df.groupby(["Concept", "Model", "Train Source", "Test Source"])
table_df_grouped.mean().reset_index().to_csv(os.path.join(RESULTS, "leace_mis_mean.csv"), index=False)
table_df_grouped.std().reset_index().to_csv(os.path.join(RESULTS, "leace_mis_std.csv"), index=False)

#%%
entropy_cols = [
    'Hz_c', 'Hz_c_mid_h', 'MIz_c_h', 
    'Hqbot_c', 'Hqbot_c_mid_hbot','MIqbot_c_hbot', 
    'Hqpar_c', 'Hqpar_c_mid_hpar', 'MIqpar_c_hpar',
    'Hz_x_c', 'Hz_x_mid_h_c', 'MIz_x_h_mid_c', 'Hqbot_x_c',
    'Hqbot_x_mid_hbot_c', 'MIqbot_x_hbot_mid_c', 
    'Hqpar_x_c',
    'Hqpar_x_mid_hpar_c', 'MIqpar_x_hpar_mid_c'
]
entropy_breakdown = df[['model_name', 'concept', 'proj_source', 'eval_source'] + entropy_cols]

entcols_name = {
    "model_name": "Model",
    "concept": "Concept",
    "proj_source": "Train Source",
    "eval_source": "Test Source",
    #"index": "Concept + Model",
    #"newindex": "Concept + Model + Metric",
    "metric": "Metric",
    'Hz_c': "Hz(C)", 
    'Hz_c_mid_h': "Hz(C | H)", 
    'MIz_c_h': "MIz(C; H)", 
    'Hqbot_c': "Hqbot(C)", 
    'Hqbot_c_mid_hbot': "Hqbot(C | Hbot)",
    'MIqbot_c_hbot': "MIqbot(C; Hbot)", 
    'Hqpar_c': "Hqpar(C)", 
    'Hqpar_c_mid_hpar': "Hqpar(C | Hpar)", 
    'MIqpar_c_hpar': "MIqpar(C; Hpar)", 
    'Hz_x_c': "Hz(X | C)",  
    'Hz_x_mid_h_c': "Hz(X | H, C)",  
    'MIz_x_h_mid_c': "MIz(X; H | C)", 
    'Hqbot_x_c': "Hqbot(X | C)",  
    'Hqbot_x_mid_hbot_c': "Hqbot(X | Hbot, C)",
    'MIqbot_x_hbot_mid_c': "MIqbot(X; Hbot | C)",
    'Hqpar_x_c': "Hqpar(X | C)",  
    'Hqpar_x_mid_hpar_c': "Hqpar(X | Hpar, C)",
    'MIqpar_x_hpar_mid_c': "MIqpar(X; Hpar | C)",
}


#entropy_breakdown.sort_values(by = ["concept", "model_name", "source"], inplace=True)
entropy_breakdown.columns = [entcols_name[x] for x in entropy_breakdown.columns]
entropy_breakdown_grouped = entropy_breakdown.groupby(["Concept", "Model", "Train Source", "Test Source"])
entropy_breakdown_grouped.mean().reset_index().to_csv(os.path.join(RESULTS, "leace_entropies_mean.csv"), index=False)
entropy_breakdown_grouped.std().reset_index().to_csv(os.path.join(RESULTS, "leace_entropies_std.csv"), index=False)

#%% CORRELATIONAL
corrfilepaths = get_res_file_paths("corr_mis", "corr_june15")

corr_res_records = []
for mifile in corrfilepaths:
    with open(mifile, 'rb') as f:      
        mires = pickle.load(f)
    corr_res_records.append(mires)

corr_df = pd.DataFrame(corr_res_records)

#%%
corr_df.groupby(["concept", "model_name", "proj_source", "eval_name"]).count().to_csv(
    os.path.join(RESULTS, "corr_counts.csv")
)

#%%
corr_table_df = corr_df[[
    'model_name', 'concept', 'proj_source', 
    #'mi_c_h', 'mi_c_hbot', 'mi_c_hpar', 
    #'reconstructed', 'encapsulation',
    #'perc_mi_c_hbot', 'perc_mi_c_hpar',
    #'perc_encapsulation', 'perc_reconstructed',
    #'cont_mi', 'stab_mi',  'ent_pxc',
    #'train_all_Hz_C', 'train_all_Hz_c_mid_hbot', 
    'train_all_MIz_c_h', 
    'train_all_MIc_c_hbot', 
    'train_concept_MIz_c_h',
    'train_concept_MIc_c_hbot',
    'test_all_MIz_c_h',
    'test_all_MIc_c_hbot',
    'test_concept_MIz_c_h',
    'test_concept_MIc_c_hbot',
]]

corr_mi_renames = {
    "model_name": "Model",
    "concept": "Concept",
    "proj_source": "Train Source",
    'train_all_MIz_c_h': "Train All I(C;H)", 
    'train_all_MIc_c_hbot': "Train All I(C;Hbot)", 
    'train_concept_MIz_c_h': "Train Concept I(C;H)",
    'train_concept_MIc_c_hbot': "Train Concept I(C;Hbot)", 
    'test_all_MIz_c_h': "Test All I(C;H)", 
    'test_all_MIc_c_hbot': "Test All I(C;Hbot)", 
    'test_concept_MIz_c_h': "Test Concept I(C;H)", 
    'test_concept_MIc_c_hbot': "Test Concept I(C;Hbot)",
}


corr_table_df.columns = [corr_mi_renames[x] for x in corr_table_df.columns]
corr_table_df_grouped = corr_table_df.groupby(["Concept", "Model", "Train Source"])
corr_table_df_grouped.mean().reset_index().to_csv(os.path.join(RESULTS, "corr_mis_mean.csv"), index=False)
corr_table_df_grouped.std().reset_index().to_csv(os.path.join(RESULTS, "corr_mis_std.csv"), index=False)


#%% 
#ent_break_group = entropy_breakdown.groupby(["source", "concept", "model_name"])

#ent_mean = ent_break_group.mean().reset_index()
#ent_mean["index"] = ent_mean["concept"] + "_" + ent_mean["model_name"]
#ent_mean.drop(["source", "concept", "model_name"], axis=1, inplace=True)
#ent_mean = ent_mean[["index"] + entropy_cols]
#ent_mean["metric"] = "mean"

#ent_std = ent_break_group.std().reset_index()
#ent_std["index"] = ent_std["concept"] + "_" + ent_std["model_name"]
#ent_std.drop(["source", "concept", "model_name"], axis=1, inplace=True)
#ent_std = ent_std[["index"] + entropy_cols]
#ent_std["metric"] = "std"
#ent_final = pd.concat([ent_mean, ent_std], axis=0)
#ent_final["newindex"] = ent_final["index"] + "_" + ent_final["metric"]
#ent_final = ent_final[["newindex"] + entropy_cols]

#ent_final.columns = [entcols_name[x] for x in ent_final.columns]
#ent_final.T.to_csv(os.path.join(RESULTS, "leace_entropies.csv"), index=True)
#.reset_index().to_csv(os.path.join(RESULTS, "leace_entropies_std.csv"), index=False)

#.reset_index().to_csv(os.path.join(RESULTS, "leace_entropies_mean.csv"), index=False)
# %%
