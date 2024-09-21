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
import pandas as pd


sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
from evals.agg_mi_evals import get_res_file_paths

int_file_paths = get_res_file_paths(RESULTS, "int", "int_june27")

#%%
int_dfs = []
for intfile in int_file_paths:
    with open(intfile, 'rb') as f:      
        intres = pd.read_csv(f, index_col=0)
    int_dfs.append(intres)

df = pd.concat(int_dfs)
df.to_csv(os.path.join(RESULTS, f"full_int_june27.csv"))

# %%
df.groupby(["concept", "model_name", "proj_source", "int_source", "eval_name"]).count().to_csv(
    os.path.join(RESULTS, f"int_counts.csv")
)

#%%
proj_source_renames = {
    "natural_all": "Curated",
    "natural_concept": "Curated",
    "gen_ancestral_all": "Gen (Ancestral)",
    "gen_nucleus_all": "Gen (Nucleus)",
}
df["proj_source"] = df["proj_source"].apply(lambda x: proj_source_renames[x])



#%%
table_df = df[[
    'model_name', 'concept', 'run_path', 'iteration',
    #'proj_source', 'int_source', 
    'y',
    'base_correct', 'corr_erased_correct',
    'erased_correct', 'do_c0_correct', 'do_c1_correct'
]]

#int_renames = {
#    "model_name": "Model",
#    "concept": "Concept",
#    "run_path": "Run Path",
#    "iteration": "Eval Iteration",
#    "proj_source": "Train Source",
#    "int_source": "Int Source",
#    'y': 'Concept Label',
#    'base_correct': "Acc. p(x|h)", 
#    'corr_erased_correct': "Acc. p(x|hbot)",
#    'erased_correct': "Acc. q(x|hbot)", 
#    'do_c0_correct': "Acc. p(x|hbot, do(C=0))", 
#    'do_c1_correct': "Acc. p(x|hbot, do(C=1))"
#}

#table_df.columns = [int_renames[x] for x in table_df.columns]

#%%
iter_maj_accs = table_df.groupby(["concept", "model_name", "run_path", "iteration"])["y"].mean().reset_index()
maj_accs = iter_maj_accs.groupby(["concept"])["y"].mean().reset_index()
maj_accs["maj_acc"] = np.where(maj_accs["y"] >= .5, maj_accs["y"], 1 - maj_accs["y"])
maj_accs.drop("y", axis=1, inplace=True)
maj_accs.to_csv(os.path.join(RESULTS, "int_maj_accs.csv"), index=False)
#table_df_grouped.mean().reset_index().to_csv(os.path.join(RESULTS, "int_mean.csv"), index=False)
#table_df_grouped.std().reset_index().to_csv(os.path.join(RESULTS, "int_std.csv"), index=False)


# %%
fig3_table = table_df[
    ["concept", "model_name", "run_path", "iteration",
    "base_correct", "corr_erased_correct", "erased_correct", 
    "do_c0_correct", "do_c1_correct"]
]
long_table_df = pd.melt(
    fig3_table,
    id_vars=["concept", "model_name", "run_path", "iteration"], 
    value_vars=["base_correct", "corr_erased_correct", "erased_correct", "do_c0_correct", "do_c1_correct"],
    var_name="raw_metric"
)
metric_mapping = {
    "base_correct": "base_correct",
    "corr_erased_correct": "corr_erased_correct",
    "erased_correct": "erased_correct",
    "do_c0_correct": "do_correct",
    "do_c1_correct": "do_correct",
}
long_table_df["metric"] = long_table_df["raw_metric"].apply(lambda x: metric_mapping[x])

long_table_df_means = long_table_df.groupby(["concept", "model_name", "run_path", "iteration", "metric"])["value"].mean().reset_index()
#table_df_grouped.drop("y", axis=1, inplace=True)

# %%
long_table_df_means.to_csv(
    os.path.join(RESULTS, "fig3_intres_input.csv"),
    index=False
)
# %%
