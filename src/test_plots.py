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
from tqdm import tqdm
import pandas as pd
import pickle
import torch
import random 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
from data.embed_wordlists.embedder import load_concept_token_lists
from utils.dataset_loaders import load_processed_data
#from evals.usage_eval import diag_eval, usage_eval

from utils.dataset_loaders import load_processed_data
from paths import OUT
from evals.kl_eval import compute_kl, renormalize, get_distribs, load_model_eval
from tqdm import trange

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#####################
# Creating Nice Graphs #
########################
acc = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/acc.csv")
mi = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/er.csv")
fth = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/fth.csv")
#%%
import matplotlib.pyplot as plt 
import seaborn as sns
acc["model_concept"] = acc["concept"] + "_" + acc["model"]
#accgen = acc[acc["concept"] == "gender"]
accplot = acc.drop(
    acc[(acc["prefix"] == "base") | (acc["prefix"] == "P")].index, axis=0
)[["model_concept", "k", "prefix", "test_accuracy"]]
accplot_mean = accplot.groupby(["model_concept", "k", "prefix"]).mean().reset_index()
accbase = acc[acc["prefix"] == "base"][["model_concept","maj_acc_test"]].groupby("model_concept").mean()

#%%
fig, axes = plt.subplots(1,4,figsize=(30,6))
pairs = [
    (axes[0], "number_gpt2-large"),
    (axes[1], "number_bert-base-uncased"),
    (axes[2], "gender_gpt2-base-french"),
    (axes[3], "gender_camembert-base"),
]
for ax, name in pairs:
    sns.lineplot(data=accplot_mean[accplot_mean["model_concept"] == name], x="k", y="test_accuracy", hue="prefix", ax=ax)
    ax.axhline(y=accbase.loc[name, "maj_acc_test"], color="g", linestyle="-")
    #ax.axhline(y=accbase.loc[name, "test_accuracy"], color="r", linestyle="-")
    ax.set_ylim(.5, 1)
    ax.set_xlim(1,769)

#%%
fth["model_concept"] = fth["concept"] + "_" + fth["model"]
fthplot = fth[fth["prefix"]!="baseline"][["model_concept", "k", "test_kl_all_merged"]]
fthplot_mean = fthplot.groupby(["model_concept", "k"]).mean().reset_index()
fthplot_mean.columns = ["model_concept", "k", "value"]
fthplot_mean["metric"] = "fth_kl"

#fthbase = fth[fth["prefix"]=="baseline"][["model_concept", "test_kl_all_merged"]].groupby("model_concept").mean()

mi["model_concept"] = mi["concept"] + "_" + mi["model"]
miplot = mi[mi["prefix"]=="I_P"][["model_concept", "k", "test_mi"]]
miplot_mean = miplot.groupby(["model_concept", "k"]).mean().reset_index()
miplot_mean.columns = ["model_concept", "k", "value"]
miplot_mean["metric"] = "mi"

comboplot = pd.concat((fthplot_mean, miplot_mean), axis=0)

#mibase = mi[mi["prefix"]=="base"][["model_concept", "test_mi"]].groupby("model_concept").mean()

#%%
fig, axes = plt.subplots(1,4,figsize=(30,6))
pairs = [
    (axes[0], "number_gpt2-large"),
    (axes[1], "number_bert-base-uncased"),
    (axes[2], "gender_gpt2-base-french"),
    (axes[3], "gender_camembert-base"),
]
for ax, name in pairs:
    sns.lineplot(data=comboplot[comboplot["model_concept"] == name], x="k", y="value", hue="metric", ax=ax)
    #ax.axhline(y=fthbase.loc[name, "test_kl_all_merged"], color="g", linestyle="-")
    #ax.axhline(y=mibase.loc[name, "test_mi"], color="r", linestyle="-")
    #ax.set_ylim(0, 1)
    ax.set_xlim(1,769)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
g = sns.FacetGrid(accplot, col="model_concept", hue="prefix")
g.map(sns.lineplot, "k", "test_accuracy")
g.refline(y=tips["tip"].median())

g.add_legend()


#%%



#%%
#def get_baseline_kls(concept, model_name, nsamples=200):

model_name = "bert-base-uncased"
concept = "number"
X, U, y, facts, foils = load_processed_data(concept, model_name)
base_path = os.path.join(OUT, f"run_output/{concept}/{model_name}")
run_output_path = os.path.join(base_path, "230624/run_bert-base-uncased_theta_k1_Plr0.003_Pms11,21,31,41,51_clflr0.003_clfms200_2023-06-25-12:28:54_0_1.pkl")
run = load_run_output(run_output_path)

idx = np.arange(0, X.shape[0])
np.random.shuffle(idx)
X_train, U_train, y_train = X[idx[:50000],:], U[idx[:50000],:], y[idx[:50000]]
X_test, U_test, y_test = run["X_test"], run["U_test"], run["y_test"]
I_P = run["output"]["I_P_burn"]

#%%
from evals.usage_eval import usage_eval
I = np.eye(I_P.shape[0])
res = usage_eval(I, "I_P", X_train, U_train, y_train, 
    X_test, U_test, y_test)
#V, l0_tl, l1_tl, _ = load_model_eval(model_name, concept)


# %%
#model_name = "gpt2-large"
#concept = "number"
#X, U, y, facts, foils = load_processed_data(concept, model_name)
#base_path = os.path.join(OUT, f"run_output/{concept}/{model_name}")
#run_output_path = os.path.join(base_path, "230624/run_camembert-base_theta_k1_Plr0.001_Pms26,51,76_clflr0.001_clfms26,51,76_2023-06-25-12:27:45_0_1.pkl")
#run = load_run_output(run_output_path)
