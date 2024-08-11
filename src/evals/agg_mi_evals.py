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

from paths import DATASETS, OUT, RESULTS, MODELS, TIANYU_RESULTS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

# %% MI RES
def get_res_file_paths(root_dir, res_dir, eval_run_name):
    resfolder = os.path.join(root_dir, f"{res_dir}/{eval_run_name}")
    resfiles = os.listdir(resfolder)
    resfilepaths = [os.path.join(resfolder, x) for x in resfiles]
    return resfilepaths

mifilepaths_1 = get_res_file_paths(TIANYU_RESULTS, "mis", "june27")

mifilepaths = mifilepaths_1 # + mifilepaths_2

#%%
res_records = []
for mifile in mifilepaths:
    with open(mifile, 'rb') as f:      
        mires = pickle.load(f)
    res_records.append(mires)

print(mifilepaths_1)
df = pd.DataFrame(res_records)

#%% Export counts to verify that all runs have been evaluated
df.groupby(["concept", "model_name", "proj_source", "eval_source", "eval_name"]).count().to_csv(
    os.path.join(RESULTS, "counts.csv")
)

#%%
#df["reconstructed"] = df["MIqbot_c_hbot"] + df["MIqpar_c_hpar"]
df["ratio_erasure"] = 1 - (df["no_na_MIqbot_c_hbot"] / df["no_na_MIz_c_h"])
df["ratio_encapsulation"] = df["no_na_MIqpar_c_hpar"] / df["no_na_MIz_c_h"]
df["no_na_reconstructed"] = df["no_na_MIqbot_c_hbot"] + df["no_na_MIqpar_c_hpar"]
df["ratio_reconstructed"] = df["no_na_reconstructed"] / df["no_na_MIz_c_h"]
df["ratio_containment"] = 1 - (df["MIqpar_x_hpar_mid_c"]/df["MIz_x_h_mid_c"])
df["ratio_stability"] = df["MIqbot_x_hbot_mid_c"]/df["MIz_x_h_mid_c"]

for k in ["ratio_erasure", "ratio_encapsulation", "ratio_reconstructed", "ratio_containment", "ratio_stability"]:
    print(k, np.mean(df[k]))

#%%
proj_source_renames = {
    "natural_all": "CEBaB",
    "gen_ancestral_all": "Gen (Ancestral)",
    "gen_nucleus_all": "Gen (Nucleus)",
    "natural_concept": "Natural Concept"
}
df["proj_source"] = df["proj_source"].apply(lambda x: proj_source_renames[x])

eval_source_renames = {
    "train_all": "Train (All)",
    "train_concept": "Train (Concept)",
    "test_all": "Test (All)",
    "test_concept": "Test (Concept)",
}
df["eval_source"] = df["eval_source"].apply(lambda x: eval_source_renames[x])


# %%
#df["sampling_method"] = df["nucleus"].apply(lambda x: np.where(x, "Nucleus", "Ancestral"))
table_df = df[[
    'model_name', 'concept', #'proj_source', 'eval_source', 
    #'mi_c_h', 'mi_c_hbot', 'mi_c_hpar', 
    #'reconstructed', 'encapsulation',
    #'perc_mi_c_hbot', 'perc_mi_c_hpar',
    #'perc_encapsulation', 'perc_reconstructed',
    #'cont_mi', 'stab_mi',  'ent_pxc',
    'no_na_MIz_c_h',
    "ratio_erasure",
    "ratio_encapsulation",
    "ratio_reconstructed",
    'MIz_x_h_mid_c',
    "ratio_containment",
    "ratio_stability",
]]

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
    'no_na_MIz_c_h': "I(C;H)",
    'MIz_x_h_mid_c': "I(X;H|C)", 
    "ratio_erasure": "Erasure Ratio",
    "ratio_encapsulation": "Encapsulation Ratio",
    "ratio_reconstructed": "Reconstructed Ratio",
    "ratio_containment": "Containment Ratio",
    "ratio_stability": "Stability Ratio",
}

table_df.sort_values(by = ["concept", "model_name"], inplace=True)
table_df.columns = [mi_renames[x] for x in table_df.columns]
counterfactual_mi_grouped = table_df.groupby(["Concept", "Model"])
counterfactual_mi_grouped_mean = counterfactual_mi_grouped.mean().reset_index()#.to_csv(os.path.join(RESULTS, "leace_mis_mean.csv"), index=False)
counterfactual_mi_grouped_std = counterfactual_mi_grouped.std().reset_index()#.to_csv(os.path.join(RESULTS, "leace_mis_std.csv"), index=False)

#%%
entropy_cols = [
    'no_na_Hz_c', 'no_na_Hz_c_mid_h', 'no_na_MIz_c_h', 
    'no_na_Hqbot_c', 'no_na_Hqbot_c_mid_hbot','no_na_MIqbot_c_hbot', 
    'no_na_Hqpar_c', 'no_na_Hqpar_c_mid_hpar', 'no_na_MIqpar_c_hpar',
    'Hz_x_c', 'Hz_x_mid_h_c', 'MIz_x_h_mid_c', 'Hqbot_x_c',
    'Hqbot_x_mid_hbot_c', 'MIqbot_x_hbot_mid_c', 
    'Hqpar_x_c',
    'Hqpar_x_mid_hpar_c', 'MIqpar_x_hpar_mid_c',
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
    'no_na_Hz_c': "Hz(C)", 
    'no_na_Hz_c_mid_h': "Hz(C | H)", 
    'no_na_MIz_c_h': "MIz(C; H)", 
    'no_na_Hqbot_c': "Hqbot(C)", 
    'no_na_Hqbot_c_mid_hbot': "Hqbot(C | Hbot)",
    'no_na_MIqbot_c_hbot': "MIqbot(C; Hbot)", 
    'no_na_Hqpar_c': "Hqpar(C)", 
    'no_na_Hqpar_c_mid_hpar': "Hqpar(C | Hpar)", 
    'no_na_MIqpar_c_hpar': "MIqpar(C; Hpar)", 
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
corrfilepaths = get_res_file_paths(TIANYU_RESULTS, "corr_mis", "corr_june27")

corr_res_records = []
for mifile in corrfilepaths:
    with open(mifile, 'rb') as f:      
        mires = pickle.load(f)
    corr_res_records.append(mires)

corr_df = pd.DataFrame(corr_res_records)

corr_df["corr_erasure_ratio"] = (
    1 - (corr_df["test_concept_MIc_c_hbot"] / corr_df["test_concept_MIz_c_h"])
)

#%%
corr_df.groupby(["concept", "model_name", "proj_source", "eval_name"]).count().to_csv(
    os.path.join(RESULTS, "corr_counts.csv")
)

#%%
corr_table_df = corr_df[[
    'model_name', 'concept', #'proj_source', 
    #'mi_c_h', 'mi_c_hbot', 'mi_c_hpar', 
    #'reconstructed', 'encapsulation',
    #'perc_mi_c_hbot', 'perc_mi_c_hpar',
    #'perc_encapsulation', 'perc_reconstructed',
    #'cont_mi', 'stab_mi',  'ent_pxc',
    #'train_all_Hz_C', 'train_all_Hz_c_mid_hbot', 
    #'train_all_MIz_c_h', 
    #'train_all_MIc_c_hbot', 
    #'train_concept_MIz_c_h',
    #'train_concept_MIc_c_hbot',
    #'test_all_MIz_c_h',
    #'test_all_MIc_c_hbot',
    #'test_concept_MIz_c_h',
    #'test_concept_MIc_c_hbot',
    'corr_erasure_ratio'
]]
#print(corr_table_df)

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
    'corr_erasure_ratio': "Correlational Erasure Ratio"
}

corr_table_df.columns = [corr_mi_renames[x] for x in corr_table_df.columns]
corr_table_df_grouped = corr_table_df.groupby(["Concept", "Model"])
corr_table_df_grouped_mean = corr_table_df_grouped.mean().reset_index()#.to_csv(os.path.join(RESULTS, "corr_mis_mean.csv"), index=False)
corr_table_df_grouped_std = corr_table_df_grouped.std().reset_index()#.to_csv(os.path.join(RESULTS, "corr_mis_std.csv"), index=False)

#%%
combined_mi_table_mean = pd.merge(
    left = counterfactual_mi_grouped_mean, 
    right = corr_table_df_grouped_mean,
    on = ["Concept", "Model"],
    how = "outer"
)
combined_mi_table_std = pd.merge(
    left = counterfactual_mi_grouped_std, 
    right = corr_table_df_grouped_std,
    on = ["Concept", "Model"],
    how = "outer"
)

column_order = ['Concept', 'Model', 'I(C;H)', 'Correlational Erasure Ratio',
    'Erasure Ratio', 'Encapsulation Ratio', 'Reconstructed Ratio',
    'I(X;H|C)', 'Containment Ratio', 'Stability Ratio',
]
combined_mi_table_mean = combined_mi_table_mean[column_order]
combined_mi_table_std = combined_mi_table_std[column_order]

#%%
combined_mi_table_mean.to_csv(os.path.join(RESULTS, "combined_mis_mean.csv"), index=False)
combined_mi_table_std.to_csv(os.path.join(RESULTS, "combined_mis_std.csv"), index=False)
#%%