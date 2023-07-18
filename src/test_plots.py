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
import matplotlib.pyplot as plt 
import seaborn as sns

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

ssfont = {'fontname':'Times New Roman'}
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

#%%#####################
# Creating Nice Graphs #
########################
res = pd.read_csv(os.path.join(RESULTS,"res.csv"))
res["model_concept"] = res["concept"] + "_" + res["model"]
res.drop(["model", "concept"], axis=1, inplace=True)
#mi = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/er.csv")
#fth = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/fth.csv")

#%%
#acc["combo_accuracy"] = np.where(
#    acc["model"].isin(["gpt2-large", "gpt2-base-french"]),
#    acc["gen_accuracy"], acc["test_accuracy"])
#accgen = acc[acc["concept"] == "gender"]
accplot = res[res["metric"].isin([
    "test_P_acc_correct", "test_I_P_acc_correct", 
    "gen_P_acc_correct", "gen_I_P_acc_correct"])]

acc_renames = {
    "test_P_acc_correct": "I-P Accuracy (Test Samples)",
    "test_I_P_acc_correct": "P Accuracy (Test Samples)",
    "test_P_acc_correct_l0": "I-P Accuracy (L0 Test Samples)",
    "test_I_P_acc_correct_l0": "P Accuracy (L0 Test Samples)",
    "test_P_acc_correct_l1": "I-P Accuracy (L1 Test Samples)",
    "test_I_P_acc_correct_l1": "P Accuracy (L1 Test Samples)",
    "gen_P_acc_correct": "I-P Accuracy (Generated Samples)",
    "gen_I_P_acc_correct": "P Accuracy (Generated Samples)",
    "gen_P_acc_correct_l0": "I-P Accuracy (L0 Generated Samples)",
    "gen_I_P_acc_correct_l0": "P Accuracy (L0 Generated Samples)",
    "gen_P_acc_correct_l1": "I-P Accuracy (L1 Generated Samples)",
    "gen_I_P_acc_correct_l1": "P Accuracy (L1 Generated Samples)",
}
accplot["clean_metric"] = [acc_renames[x] for x in accplot["metric"]]

#%%
#accplot_mean = accplot.groupby(["model_concept", "k", "prefix"]).mean().reset_index()

#accplot_mean_gen = accplot[accplot["model_concept"].isin(
#    ["gender_gpt2-base-french", "number_gpt2-large"])].drop(
#        ["prefix", "test_accuracy"], axis=1
#    )
#accplot_mean_gen["metric"] = "Accuracy (Generated Samples)"
#accplot_mean_gen.columns = ["model_concept", "k", "value", "metric"]

#accplot_mean_test = accplot.drop(["gen_accuracy", "prefix"], axis=1)
#accplot_mean_test["metric"] = "Accuracy (Test Samples)"
#accplot_mean_test.columns = ["model_concept", "k", "value", "metric"]

#accplot_mean_combo = pd.concat([accplot_mean_gen, accplot_mean_test], axis=0)

#accbase = acc[acc["prefix"] == "base"][["model_concept","maj_acc_test"]].groupby("model_concept").mean()

#%%


miplot = res[res["metric"].isin(
    ['test_P_fth_mi_l0', 'test_P_fth_mi_l1', 'test_I_P_fth_mi_l0', 'test_I_P_fth_mi_l1',    
    'test_base_mi', 'test_I_P_mi', 
    'test_reconstructed_info', 'test_encapsulation', 'test_perc_reconstructed', 
    'gen_P_fth_mi_l0', 'gen_I_P_fth_mi_l0', 'gen_P_fth_mi_l1', 'gen_I_P_fth_mi_l1', 
    'gen_base_mi', 'gen_I_P_mi',
    'gen_reconstructed_info', 'gen_encapsulation', 'gen_perc_reconstructed'
    ])]
#"test_P_acc_correct_l0", "test_I_P_acc_correct_l0", "test_P_acc_correct_l1", "test_I_P_acc_correct_l1",
#"gen_P_acc_correct_l0", "gen_I_P_acc_correct_l0", "gen_P_acc_correct_l1", "gen_I_P_acc_correct_l1"
mi_renames = {
    "test_P_fth_mi_l0": "I-P Stability (L0 Test Samples)",
    "test_I_P_fth_mi_l0": "P Stability (L0 Test Samples)",
    "test_P_fth_mi_l1": "I-P Stability (L1 Test Samples)",
    "test_I_P_fth_mi_l1": "P Stability (L1 Test Samples)",
    "test_base_mi": "Total Info (Test Samples)",
    "test_encapsulation": "Encapsulation (Test Samples)",
    "test_reconstructed_info": "Reconstructed Info (Test Samples",
    "test_perc_reconstructed": "Reconstructed % (Test Samples)",
    "test_I_P_mi": "P Erasure MI (Test Samples)",
    "gen_P_fth_mi_l0": "I-P Stability (L0 Gen Samples)",
    "gen_I_P_fth_mi_l0": "P Stability (L0 Gen Samples)",
    "gen_P_fth_mi_l1": "I-P Stability (L1 Gen Samples)",
    "gen_I_P_fth_mi_l1": "P Stability (L1 Gen Samples)",
    "gen_base_mi": "Total Info (Gen Samples)",
    "gen_encapsulation": "Encapsulation (Gen Samples)",
    "gen_I_P_mi": "P Erasure MI (Gen Samples)",
    "gen_reconstructed_info": "Reconstructed Info (Gen Samples",
    "gen_perc_reconstructed": "Reconstructed % (Gen Samples)",
}
miplot["clean_metric"] = [mi_renames[x] for x in miplot["metric"]]

#%%
resfilter = res[(res["metric"].isin(
    ["test_P_fth_mi_l0", "test_I_P_fth_mi_l0", "test_P_fth_mi_l1", 
     "test_I_P_fth_mi_l1", "test_I_P_mi", "gen_P_fth_mi_l0", 
     "gen_I_P_fth_mi_l0", "gen_P_fth_mi_l1", "gen_I_P_fth_mi_l1", "gen_I_P_mi"])) &
    (res["k"].isin([0, 1, 2]))]
resmean = resfilter.groupby(["model_concept", "k", "metric"])["value"].mean().reset_index()
respivot = pd.pivot(resmean, index=["model_concept", "k"], columns="metric", values="value").reset_index()
respivot = respivot[
    ["model_concept", "k", "test_I_P_mi", "gen_I_P_mi",
     "test_P_fth_mi_l0", "test_P_fth_mi_l1", "test_I_P_fth_mi_l0", "test_I_P_fth_mi_l1", 
     "gen_P_fth_mi_l0", "gen_P_fth_mi_l1","gen_I_P_fth_mi_l0", "gen_I_P_fth_mi_l1"]]
column_renames = mi_renames
column_renames["model_concept"] = "Concept & Model"
respivot.columns = [mi_renames[x] if x not in ["k"] else x for x in respivot.columns]
respivot.to_csv(os.path.join(RESULTS, "erasure_stab.csv"), index=False)

#%%
"""
fth["model_concept"] = fth["concept"] + "_" + fth["model"]
fthplot = fth[fth["prefix"]!="baseline"][["model_concept", "k", "test_kl_all_merged", "gen_kl_all_merged"]]
#fthplot_mean = fthplot.groupby(["model_concept", "k"]).mean().reset_index()

fthplot_mean_gen = fthplot[fthplot["model_concept"].isin(
    ["gender_gpt2-base-french", "number_gpt2-large"])].drop(
        ["test_kl_all_merged"], axis=1
    )
fthplot_mean_gen["metric"] = "$(\mathbf{I}-\mathbf{P})\,\mathrm{D}_{\mathrm{KL}}$ (Generated Samples)"
fthplot_mean_gen.columns = ["model_concept", "k", "bits", "metric"]

fthplot_mean_test = fthplot.drop(["gen_kl_all_merged"], axis=1)
fthplot_mean_test["metric"] = "$(\mathbf{I}-\mathbf{P})\,\mathrm{D}_{\mathrm{KL}}$ (Test Samples)"
fthplot_mean_test.columns = ["model_concept", "k", "bits", "metric"]

fthplot_mean_combo = pd.concat([fthplot_mean_gen, fthplot_mean_test], axis=0)

#fthbase = fth[fth["prefix"]=="baseline"][["model_concept", "test_kl_all_merged"]].groupby("model_concept").mean()

#%%
mi["model_concept"] = mi["concept"] + "_" + mi["model"]
miplot = mi[mi["prefix"].isin(["I_P"])][["model_concept", "k", "test_mi", "gen_mi"]]
#miplot_mean = miplot.groupby(["model_concept", "k"]).mean().reset_index()

miplot_mean_gen = miplot[miplot["model_concept"].isin(
    ["gender_gpt2-base-french", "number_gpt2-large"])].drop(
        ["test_mi"], axis=1
    )
miplot_mean_gen["metric"] = "$(\mathbf{I}-\mathbf{P})\,\mathrm{I}$ (Generated Samples)"
miplot_mean_gen.columns = ["model_concept", "k", "bits", "metric"]

miplot_mean_test = miplot.drop(["gen_mi"], axis=1)
miplot_mean_test["metric"] = "$(\mathbf{I}-\mathbf{P})\,\mathrm{CE}$ (Test Samples)"
miplot_mean_test.columns = ["model_concept", "k", "bits", "metric"]

miplot_mean_combo = pd.concat([miplot_mean_gen, miplot_mean_test], axis=0)

mifth_comboplot = pd.concat((fthplot_mean_combo, miplot_mean_combo), axis=0)

#mibase = mi[mi["prefix"]=="base"][["model_concept", "test_mi"]].groupby("model_concept").mean()
"""
#%%
acc_renames_values = [x for x in acc_renames.values()]
mi_renames_values = [x for x in mi_renames.values()]
all_keys = acc_renames_values + mi_renames_values
cols = sns.color_palette("bright", len(all_keys))
palette = {}
for k, v in zip(all_keys, cols):
    palette[k] = v

#%%
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]
fig, axes = plt.subplots(4,4,figsize=(40,24))
#fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.05, hspace=0.05)


pairs = [
    (axes[0,0], "number_gpt2-large", "acc"),
    (axes[0,1], "number_bert-base-uncased", "acc"),
    (axes[0,2], "gender_gpt2-base-french", "acc"),
    (axes[0,3], "gender_camembert-base", "acc"),
    (axes[1,0], "number_gpt2-large", "er_test"),
    (axes[1,1], "number_bert-base-uncased", "er_test"),
    (axes[1,2], "gender_gpt2-base-french", "er_test"),
    (axes[1,3], "gender_camembert-base", "er_test"),
    (axes[2,0], "number_gpt2-large", "er_gen"),
    (axes[2,1], "number_bert-base-uncased", "er_gen"),
    (axes[2,2], "gender_gpt2-base-french", "er_gen"),
    (axes[2,3], "gender_camembert-base", "er_gen"),
    (axes[3,0], "number_gpt2-large", "fth"),
    (axes[3,1], "number_bert-base-uncased", "fth"),
    (axes[3,2], "gender_gpt2-base-french", "fth"),
    (axes[3,3], "gender_camembert-base", "fth"),
]
for i, (ax, name, case) in enumerate(pairs):
    ax.tick_params(labelsize="xx-large")
    if "bert" in name:
        ax.set_xlim(-10,769)
        #ax.set_xlim(0,4)
    elif "gpt2-base-french" in name:
        ax.set_xlim(-10,768)
        #ax.set_xlim(0,4)
    else:
        ax.set_xlim(-10,1280)
        #ax.set_xlim(0,4)
    if case == "acc":
        accplotdf = accplot[accplot["model_concept"] == name]
        sns.lineplot(
            data=accplotdf, x="k", y="value", hue="clean_metric", ci = "sd", palette=palette, 
            ax=ax
        )
        #ax.axhline(y=accbase.loc[name, "maj_acc_test"], color="green", linestyle="dashed", label="Majority Accuracy")
        #ax.axhline(y=accbase.loc[name, "test_accuracy"], color="r", linestyle="-")
        ax.set_ylim(.5, 1)
        ax.set_ylabel("Accuracy")
        
        namesplit = name.split("_")
        concept, model = namesplit[0], namesplit[1]
        ax.set_title(f"{concept[0].upper() + concept[1:]}, {model}", fontsize=25)

        #removing legend titles
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
        #ax.get_shared_x_axes().join(ax, axes[1,i])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.legend(fontsize="xx-large")
        ax.set_ylabel("Accuracy", fontsize=20)
        if i!=0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        #, **ssfont)
    elif case == "er_test":
        #midfplot = miplot[
        #    (miplot["model_concept"] == name) & 
        #    (miplot["metric"].isin(
        #        ['test_base_mi', 'test_I_P_mi', 
        #        'test_reconstructed_info', 'test_encapsulation', 'test_perc_reconstructed', 
        #        'gen_base_mi', 'gen_I_P_mi',
        #        'gen_reconstructed_info', 'gen_encapsulation', 'gen_perc_reconstructed']))
        #]
        #sns.lineplot(data=midfplot, x="k", y="value", hue="clean_metric", ci="sd", palette=palette, ax=ax)
        ax2 = ax.twinx()
        y1df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                ['test_base_mi', 'test_I_P_mi','test_reconstructed_info', 'test_encapsulation',
                #'gen_base_mi', 'gen_I_P_mi', 'gen_reconstructed_info', 'gen_encapsulation',
                ]))
        ]
        sns.lineplot(data=y1df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax)
        y2df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(['test_perc_reconstructed']))]
        sns.lineplot(data=y2df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax2)
        ax2.set_ylim(0,2)
        #ax.axhline(y=fthbase.loc[name, "test_kl_all_merged"], color="g", linestyle="-")
        #ax.axhline(y=mibase.loc[name, "test_mi"], color="r", linestyle="-")
        #ax.set_ylim(0, 1)
        #ax.set_title(name)#, **ssfont)
        ax.set_ylabel("I (Test Samples)", fontsize=18)#, **ssfont)
        #ax2.set_ylabel("MI/CE Bits", **ssfont)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
        #ax.legend(fontsize="xx-large")#.fontsize = "x-large"
        ax.legend(fontsize="large")
        ax2.legend().set_title("")
        ax2.legend(fontsize="large", loc="upper center")
        ax.set_xticklabels([])
        ax.set_xlabel("$k$", fontsize=20)#, **ssfont)
        #ax.set_ylim(-.25,9)
        #ax.yticks(fontsize=20)
        if i!=4:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    elif case == "er_gen":
        if "bert" in name:
            continue
        #midfplot = miplot[
        #    (miplot["model_concept"] == name) & 
        #    (miplot["metric"].isin(
        #        ['test_base_mi', 'test_I_P_mi', 
        #        'test_reconstructed_info', 'test_encapsulation', 'test_perc_reconstructed', 
        #        'gen_base_mi', 'gen_I_P_mi',
        #        'gen_reconstructed_info', 'gen_encapsulation', 'gen_perc_reconstructed']))
        #]
        #sns.lineplot(data=midfplot, x="k", y="value", hue="clean_metric", ci="sd", palette=palette, ax=ax)
        ax2 = ax.twinx()
        y1df = miplot[
            (miplot["k"] != 0) &
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                [#'test_base_mi', 'test_I_P_mi','test_reconstructed_info', 'test_encapsulation',
                'gen_base_mi', 'gen_I_P_mi', 'gen_reconstructed_info', 'gen_encapsulation',
                ]))
        ]
        sns.lineplot(data=y1df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette,ax=ax)
        y2df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(['gen_perc_reconstructed']))]
        sns.lineplot(data=y2df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette,ax=ax2)
        ax2.set_ylim(0,2)
        #ax.axhline(y=fthbase.loc[name, "test_kl_all_merged"], color="g", linestyle="-")
        #ax.axhline(y=mibase.loc[name, "test_mi"], color="r", linestyle="-")
        #ax.set_ylim(0, 1)
        #ax.set_title(name)#, **ssfont)
        ax.set_ylabel("I (Gen Samples)", fontsize=18)#, **ssfont)
        #ax2.set_ylabel("MI/CE Bits", **ssfont)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        #ax.legend(fontsize="xx-large")#.fontsize = "x-large"
        ax.legend().set_title("")
        ax.legend(fontsize="large")
        ax2.legend().set_title("")
        ax2.legend(fontsize="large", loc="upper center")
        ax.set_xlabel("$k$", fontsize=20)#, **ssfont)
        ax.set_xticklabels([])
        #ax.set_ylim(-.25,9)
        #ax.yticks(fontsize=20)
        if i!=8:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    elif case == "fth":
        #ax2 = ax.twinx()
        #y1df = mifth_comboplot[(mifth_comboplot["model_concept"] == name) & 
        #                        (mifth_comboplot["metric"].isin(["I_P_gen_kl", "I_P_test_kl"]))]
        fthdfplot = miplot[
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                ["test_P_fth_mi_l0", "test_I_P_fth_mi_l0", "test_P_fth_mi_l1", 
                "test_I_P_fth_mi_l1", "gen_P_fth_mi_l0", "gen_I_P_fth_mi_l0", 
                "gen_P_fth_mi_l1", "gen_I_P_fth_mi_l1"]))
        ]
        sns.lineplot(data=fthdfplot, x="k", y="value", hue="clean_metric", ci="sd", palette=palette, ax=ax)
        #y2df = mifth_comboplot[(mifth_comboplot["model_concept"] == name) & 
        #                        (mifth_comboplot["metric"].isin(["I_P_gen_mi", "I_P_test_ce"]))]
        #sns.lineplot(data=y2df, x="k", y="bits", hue="metric", ax=ax2)
        #ax2.set_ylim(-.5,1)
        #ax.axhline(y=fthbase.loc[name, "test_kl_all_merged"], color="g", linestyle="-")
        #ax.axhline(y=mibase.loc[name, "test_mi"], color="r", linestyle="-")
        #ax.set_ylim(0, 1)
        #ax.set_title(name)#, **ssfont)
        ax.set_ylabel("Stability (Bits)", fontsize=20)#, **ssfont)
        #ax2.set_ylabel("MI/CE Bits", **ssfont)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
        #ax.legend(fontsize="xx-large")#.fontsize = "x-large"
        ax.legend(fontsize="small")#.fontsize = "x-large"
        ax.set_xlabel("$k$", fontsize=20)#, **ssfont)
        #ax.set_xticklabels([])
        #ax.set_ylim(-.25,9)
        #ax.yticks(fontsize=20)
        if i!=12:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    else:
        raise ValueError("Incorrect case specification")
        
    
    

figpath = os.path.join(RESULTS, "accfthmiplot.png")
fig.savefig(figpath)

#%%
accplottest = res[
    (res["metric"].isin(["test_I_P_acc_correct", 'test_I_P_mi'])) & 
    (res["k"].isin([0,1]))
]
accplottestmean = accplottest.groupby(["model_concept", "k", "metric"]).mean().reset_index()
accplottestpiv = pd.pivot(accplottestmean, index=["model_concept", "k"], columns=["metric"], values="value").reset_index()

genderplot = accplottestpiv[accplottestpiv["model_concept"].str.startswith("number")]
sns.scatterplot(x = accplottestpiv["test_I_P_acc_correct"], y = accplottestpiv["test_I_P_mi"])
#%%
"""
#import seaborn as sns
#import matplotlib.pyplot as plt

#sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
#g = sns.FacetGrid(accplot, col="model_concept", hue="prefix")
#g.map(sns.lineplot, "k", "test_accuracy")
#g.refline(y=tips["tip"].median())

#g.add_legend()


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


#%%
import matplotlib.gridspec as gridspec

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]
plt.figure(figsize=(2,4))
gs1 = gridspec.GridSpec(2,4)
gs1.update(wspace=.1, hspace=.2)

#fig, axes = plt.subplots(2,4,figsize=(40,16))
#fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
#fig.subplots_adjust(wspace=0.4, hspace=0.7)
pairs = [
    (axes[0,0], "number_gpt2-large", "acc"),
    (axes[0,1], "number_bert-base-uncased", "acc"),
    (axes[0,2], "gender_gpt2-base-french", "acc"),
    (axes[0,3], "gender_camembert-base", "acc"),
    (axes[1,0], "number_gpt2-large", "bits"),
    (axes[1,1], "number_bert-base-uncased", "bits"),
    (axes[1,2], "gender_gpt2-base-french", "bits"),
    (axes[1,3], "gender_camembert-base", "bits"),
]
for i, (_, name, case) in enumerate(pairs):
    ax = plt.subplots(gs1[i])
    plt.axis('on')
    if case == "acc":
        accplotdf = accplot_mean_combo[accplot_mean_combo["model_concept"] == name]
        sns.lineplot(
            data=accplotdf, x="k", y="value", hue="metric", ci = "sd", palette=palette, 
            ax=ax
        )
        ax.axhline(y=accbase.loc[name, "maj_acc_test"], color="green", linestyle="-")
        #ax.axhline(y=accbase.loc[name, "test_accuracy"], color="r", linestyle="-")
        ax.set_ylim(.5, 1)
        ax.set_ylabel("Accuracy")
        if "bert" in name:
            ax.set_xlim(-1,769)
        elif "gpt2-base-french" in name:
            ax.set_xlim(-1,768)
        else:
            ax.set_xlim(-1,1280)
        namesplit = name.split("_")
        concept, model = namesplit[0], namesplit[1]
        ax.set_title(f"{concept[0].upper() + concept[1:]}, {model}")

        #removing legend titles
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
        #, **ssfont)
    else:
        #ax2 = ax.twinx()
        #y1df = mifth_comboplot[(mifth_comboplot["model_concept"] == name) & 
        #                        (mifth_comboplot["metric"].isin(["I_P_gen_kl", "I_P_test_kl"]))]
        ydf = mifth_comboplot[(mifth_comboplot["model_concept"] == name)]
        sns.lineplot(data=ydf, x="k", y="bits", hue="metric", ci="sd", palette=palette, ax=ax)
        #y2df = mifth_comboplot[(mifth_comboplot["model_concept"] == name) & 
        #                        (mifth_comboplot["metric"].isin(["I_P_gen_mi", "I_P_test_ce"]))]
        #sns.lineplot(data=y2df, x="k", y="bits", hue="metric", ax=ax2)
        #ax2.set_ylim(-.5,1)
        #ax.axhline(y=fthbase.loc[name, "test_kl_all_merged"], color="g", linestyle="-")
        #ax.axhline(y=mibase.loc[name, "test_mi"], color="r", linestyle="-")
        #ax.set_ylim(0, 1)
        if "bert" in name:
            ax.set_xlim(-1,769)
        elif "gpt2-base-french" in name:
            ax.set_xlim(-1,768)
        else:
            ax.set_xlim(-1,1280)
        #ax.set_title(name)#, **ssfont)
        ax.set_ylabel("Bits")#, **ssfont)
        #ax2.set_ylabel("MI/CE Bits", **ssfont)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
    
    ax.set_xlabel("$k$")#, **ssfont)

figpath = os.path.join(RESULTS, "accfthmiplot.png")
fig.savefig(figpath)
"""