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
base_res = pd.read_csv(os.path.join(RESULTS,"res.csv"))
base_res.drop("maj_acc_test", axis=1, inplace=True)
p_x_res = pd.read_csv(os.path.join(RESULTS,"old_p_x_res.csv"))
p_x_res.columns = ['metric', 'value', 'model', 'concept', 'k']
res = pd.concat((base_res, p_x_res),axis=0)
res.to_csv(os.path.join(RESULTS, "full_res.csv"))

#%%
res["model_concept"] = res["concept"] + "_" + res["model"]
res.drop(["model", "concept"], axis=1, inplace=True)
#mi = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/er.csv")
#fth = pd.read_csv("/cluster/work/cotterell/cguerner/usagebasedprobing/results/fth.csv")

#%%
#badpoints = res[(res["model_concept"] == "number_gpt2-large") & (res["k"] > 1100)]
#badpoints.drop(
#    res[(res["model_concept"] == "number_gpt2-large") & (res["k"].isin([1264, 1272, 1276, 1278]))].index, axis=0, inplace=True
#)
#%%
#acc["combo_accuracy"] = np.where(
#    acc["model"].isin(["gpt2-large", "gpt2-base-french"]),
#    acc["gen_accuracy"], acc["test_accuracy"])
#accgen = acc[acc["concept"] == "gender"]
accplot = res[res["metric"].isin([
    "test_P_acc_correct", "test_I_P_acc_correct", "test_I_P_acc_correct_l0", "test_I_P_acc_correct_l1",
    "gen_P_acc_correct", "gen_I_P_acc_correct", "gen_I_P_acc_correct_l0", "gen_I_P_acc_correct_l1",
    "nucgen_P_acc_correct", "nucgen_I_P_acc_correct", "nucgen_I_P_acc_correct_l0", "nucgen_I_P_acc_correct_l1",
])]

acc_renames = {
    "test_P_acc_correct": "I-P Accuracy",
    "test_I_P_acc_correct": "P Accuracy",
    "test_P_acc_correct_l0": "I-P Accuracy (L0)",
    "test_I_P_acc_correct_l0": "P Accuracy (L0)",
    "test_P_acc_correct_l1": "I-P Accuracy (L1)",
    "test_I_P_acc_correct_l1": "P Accuracy (L1)",
    "gen_P_acc_correct": "I-P Accuracy",
    "gen_I_P_acc_correct": "P Accuracy",
    "gen_P_acc_correct_l0": "I-P Accuracy (L0)",
    "gen_I_P_acc_correct_l0": "P Accuracy (L0)",
    "gen_P_acc_correct_l1": "I-P Accuracy (L1)",
    "gen_I_P_acc_correct_l1": "P Accuracy (L1)",
    "nucgen_P_acc_correct": "I-P Accuracy",
    "nucgen_I_P_acc_correct": "P Accuracy",
    "nucgen_P_acc_correct_l0": "I-P Accuracy (L0)",
    "nucgen_I_P_acc_correct_l0": "P Accuracy (L0)",
    "nucgen_P_acc_correct_l1": "I-P Accuracy (L1)",
    "nucgen_I_P_acc_correct_l1": "P Accuracy (L1)",
}
accplot["clean_metric"] = [acc_renames[x] for x in accplot["metric"]]

#%%
miplot_basemi_prep = res[res["metric"].isin(['test_base_mi', 'gen_base_mi', 'nucgen_base_mi'])]
miplot_basemi = miplot_basemi_prep.groupby(["model_concept", "metric"])["value"].mean().reset_index()
#miplot_basemi.groupby(["model_concept", "metric"])["value"].std()
miplot_basemi = pd.pivot(miplot_basemi, index="model_concept", columns="metric", values="value")
#%%
miplot = res[res["metric"].isin(
    [#'test_P_fth_mi_l0', 'test_P_fth_mi_l1', 
    'test_I_P_fth_mi_l0', 'test_I_P_fth_mi_l1',    
    'test_I_P_mi', 'test_reconstructed', 'test_encapsulation', 
    "test_perc_I_P_mi", 'test_perc_reconstructed', "test_perc_encapsulation",
    #'gen_P_fth_mi_l0', 'gen_P_fth_mi_l1', 
    'gen_mi_x_Ph_l0', 'gen_mi_x_Ph_l1', 
    #'gen_mi_x_I_Ph_l0', 'gen_mi_x_I_Ph_l1',
    'gen_I_P_fth_mi_l0', 'gen_I_P_fth_mi_l1', 
    'gen_I_P_mi', 'gen_reconstructed', 'gen_encapsulation', 
    "gen_perc_I_P_mi", 'gen_perc_reconstructed', "gen_perc_encapsulation",
    #'gen_P_fth_mi_l0', 'gen_P_fth_mi_l1', 
    'nucgen_I_P_fth_mi_l0', 'nucgen_I_P_fth_mi_l1', 
    'nucgen_mi_x_Ph_l0', 'nucgen_mi_x_Ph_l1', 
    #'nucgen_mi_x_I_Ph_l0', 'nucgen_mi_x_I_Ph_l1',
    'nucgen_I_P_mi', 'nucgen_reconstructed', 'nucgen_encapsulation', 
    "nucgen_perc_I_P_mi", 'nucgen_perc_reconstructed', "nucgen_perc_encapsulation",
    ])]

#gen_metrics = [
    #'gen_I_P_fth_mi_l0', 'gen_I_P_fth_mi_l1', 
#    'gen_I_P_mi',
#    'gen_reconstructed', 'gen_encapsulation', 'gen_perc_reconstructed']
#test_metrics = [
    #'test_I_P_fth_mi_l0', 'test_I_P_fth_mi_l1',    
#    'test_I_P_mi', 
#    'test_reconstructed', 'test_encapsulation', 'test_perc_reconstructed']

mi_renames = {
    #"test_P_fth_mi_l0": "Stability C=0",
    "test_I_P_fth_mi_l0": "Stability C=0",
    #"test_P_fth_mi_l1": "Stability C=1",
    "test_I_P_fth_mi_l1": "Stability C=1",
    "test_base_mi": "Total MI",
    "test_encapsulation": "Encapsulation",
    "test_reconstructed": "Reconstructed MI",
    "test_I_P_mi": "Erasure",
    "test_perc_reconstructed": "Reconstructed %",
    "test_perc_I_P_mi":  "Erased %",
    "test_perc_encapsulation": "Encapsulated %", 
    #"gen_P_fth_mi_l0": "Stability C=0",
    "gen_I_P_fth_mi_l0": "Stability C=0",
    #"gen_P_fth_mi_l1": "Stability C=1",
    "gen_I_P_fth_mi_l1": "Stability C=1",
    "gen_base_mi": "Total MI",
    "gen_encapsulation": "Encapsulation",
    "gen_I_P_mi": "Erasure",
    "gen_reconstructed": "Reconstructed MI",
    "gen_perc_reconstructed": "Reconstructed %",
    "gen_perc_I_P_mi": "Erased %",
    "gen_perc_encapsulation": "Encapsulated %", 
    'gen_mi_x_Ph_l0': "Containment C=0",
    'gen_mi_x_Ph_l1': "Containment C=1",
    #"nucgen_P_fth_mi_l0": "Stability C=0",
    "nucgen_I_P_fth_mi_l0": "Stability C=0",
    #"nucgen_P_fth_mi_l1": "Stability C=1",
    "nucgen_I_P_fth_mi_l1": "Stability C=1",
    "nucgen_base_mi": "Total MI",
    "nucgen_encapsulation": "Encapsulation",
    "nucgen_I_P_mi": "Erasure",
    "nucgen_reconstructed": "Reconstructed MI",
    "nucgen_perc_reconstructed": "Reconstructed %",
    "nucgen_perc_I_P_mi": "Erased %",
    "nucgen_perc_encapsulation": "Encapsulated %", 
    'nucgen_mi_x_Ph_l0': "Containment C=0",
    'nucgen_mi_x_Ph_l1': "Containment C=1",
}
miplot["clean_metric"] = [mi_renames[x] for x in miplot["metric"]]

#%%
acc_renames_values = list(set(acc_renames.values()))
mi_renames_values = list(set(mi_renames.values()))
all_keys = acc_renames_values + mi_renames_values
cols = sns.color_palette("bright", len(all_keys))
palette = {}
for k, v in zip(all_keys, cols):
    palette[k] = v

#%%
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]
putype = "test"
from matplotlib import ticker

try:
    fig.clf()
except NameError:
    logging.info("First figure creation")


fig, axes = plt.subplots(3,4,
    gridspec_kw=dict(left=0.08, right=0.80,
                    bottom=0.12, top=0.9),
    sharey="row",
    sharex="col",
    dpi=300,
    figsize=(8,4)
)
#fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.05, hspace=0.05)


pairs = [
    #(axes[0,0], "number_gpt2-large", "acc"),
    #(axes[0,1], "number_bert-base-uncased", "acc"),
    #(axes[0,2], "gender_gpt2-base-french", "acc"),
    #(axes[0,3], "gender_camembert-base", "acc"),
    (axes[0,0], "number_gpt2-large", "er"),
    (axes[0,1], "number_bert-base-uncased", "er"),
    (axes[0,2], "gender_gpt2-base-french", "er"),
    (axes[0,3], "gender_camembert-base", "er"),
    (axes[1,0], "number_gpt2-large", "er_perc"),
    (axes[1,1], "number_bert-base-uncased", "er_perc"),
    (axes[1,2], "gender_gpt2-base-french", "er_perc"),
    (axes[1,3], "gender_camembert-base", "er_perc"),
    #(axes[2,0], "number_gpt2-large", "er_gen"),
    #(axes[2,1], "number_bert-base-uncased", "er_gen"),
    #(axes[2,2], "gender_gpt2-base-french", "er_gen"),
    #(axes[2,3], "gender_camembert-base", "er_gen"),
    (axes[2,0], "number_gpt2-large", "fth"),
    (axes[2,1], "number_bert-base-uncased", "fth"),
    (axes[2,2], "gender_gpt2-base-french", "fth"),
    (axes[2,3], "gender_camembert-base", "fth"),
]
for i, (ax, name, case) in enumerate(pairs):
    ax.tick_params(labelsize="medium")
    if "bert" in name:
        ax.set_xlim(-10,769)
        #ax.set_xlim(0,4)
    elif "gpt2-base-french" in name:
        ax.set_xlim(-10,768)
        #ax.set_xlim(0,4)
    else:
        ax.set_xlim(-10,1280)
        #ax.set_xlim(0,4)
    if case == "er":
        # Base MI line
        ax.axhline(
            y=miplot_basemi.loc[name, f"{putype}_base_mi"], 
            color=palette[mi_renames[f"{putype}_base_mi"]], 
            linestyle="-",
            label= mi_renames[f"{putype}_base_mi"],
        )
        
        y1df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin([
                f'{putype}_I_P_mi', f'{putype}_reconstructed', f'{putype}_encapsulation'
            ]))
        ]
        sns.lineplot(
            data=y1df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax
        )
        
        
        # Title
        namesplit = name.split("_")
        concept, model = namesplit[0], namesplit[1]
        ax.set_title(f"{concept[0].upper() + concept[1:]},\n {model}", fontsize="medium")
        
        # Y axis
        ax.set_ylabel("Information (Bits)", fontsize="small")#, **ssfont)
        
        # X axis
        #ax.set_xticklabels([])
        #ax.set_xlabel("")
        #ax.set_xlabel("$k$", fontsize=20)#, **ssfont)
        
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        #if i!=0:
            #ax.set_ylabel("")
            #ax.set_yticklabels([])
            #ax.set_yticks([])
        if i!=3:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )       
    ## SECOND ROW
    elif case == "er_perc":
        y2df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin([
                f'{putype}_perc_I_P_mi', f'{putype}_perc_reconstructed', f'{putype}_perc_encapsulation'
            ]))
        ]
        sns.lineplot(
            data=y2df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax
        )
        
        # Base MI line
        #ax.axhline(
        #    y=miplot_basemi.loc[name, f"{putype}_base_mi"], 
        #    color=palette[mi_renames[f"{putype}_base_mi"]], 
        #    linestyle="-",
        #    label= mi_renames[f"{putype}_base_mi"],
        #)
        
        # Title
        #namesplit = name.split("_")
        #concept, model = namesplit[0], namesplit[1]
        #ax.set_title(f"{concept[0].upper() + concept[1:]},\n {model}", fontsize="medium")
        
        # X axis
        #ax.set_xticklabels([])
        #ax.set_xlabel("")

        # Y axis
        ax.set_ylabel("% Total MI", fontsize="small")#, **ssfont)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=None, symbol=None))
        #ax.yaxis.set_tick_params()
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        #if i!=4:
        #    ax.set_ylabel("")
        #    ax.set_yticklabels([])
        #    ax.set_yticks([])
        if i!=7:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )
    ### THIRD ROW          
    elif case == "fth":
        
        fthdfplot = miplot[
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                [f"{putype}_I_P_fth_mi_l0", f"{putype}_I_P_fth_mi_l1"]))
        ]
        sns.lineplot(
            data=fthdfplot, x="k", y="value", hue="clean_metric", 
            ci="sd", palette=palette, ax=ax
        )
        
        ax.set_ylabel("Stability (Bits)", fontsize="small")#, **ssfont)
        
        #if i!=8:
        #    ax.set_ylabel("")
        #    ax.set_yticklabels([])
        if i!=11:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )        
            
    else:
        raise ValueError(f"Incorrect case specification {case}")
        
    

figpath = os.path.join(RESULTS, f"accfthmiplot_{putype}.png")
fig.savefig(figpath)

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(figpath[:-4]+".pdf") as pp:
    pp.savefig(fig)

#%%
from matplotlib import ticker

try:
    fig.clf()
except NameError:
    logging.info("First figure creation")


fig, axes = plt.subplots(3,4,
    gridspec_kw=dict(left=0.08, right=0.80,
                    bottom=0.12, top=0.9),
    sharey="row",
    sharex="col",
    dpi=300,
    figsize=(8,4)
)
#fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.05, hspace=0.05)


pairs = [
    #(axes[0,0], "number_gpt2-large", "acc"),
    #(axes[0,1], "number_bert-base-uncased", "acc"),
    #(axes[0,2], "gender_gpt2-base-french", "acc"),
    #(axes[0,3], "gender_camembert-base", "acc"),
    (axes[0,0], "number_gpt2-large", "er", "gen"),
    (axes[0,1], "number_gpt2-large", "er", "nucgen"),
    (axes[0,2], "gender_gpt2-base-french", "er", "gen"),
    (axes[0,3], "gender_gpt2-base-french", "er", "nucgen"),
    (axes[1,0], "number_gpt2-large", "er_perc", "gen"),
    (axes[1,1], "number_gpt2-large", "er_perc", "nucgen"),
    (axes[1,2], "gender_gpt2-base-french", "er_perc", "gen"),
    (axes[1,3], "gender_gpt2-base-french", "er_perc", "nucgen"),
    #(axes[2,0], "number_gpt2-large", "er_gen"),
    #(axes[2,1], "number_bert-base-uncased", "er_gen"),
    #(axes[2,2], "gender_gpt2-base-french", "er_gen"),
    #(axes[2,3], "gender_camembert-base", "er_gen"),
    (axes[2,0], "number_gpt2-large", "fth", "gen"),
    (axes[2,1], "number_gpt2-large", "fth", "nucgen"),
    (axes[2,2], "gender_gpt2-base-french", "fth", "gen"),
    (axes[2,3], "gender_gpt2-base-french", "fth", "nucgen"),
]
for i, (ax, name, case, putype) in enumerate(pairs):
    ax.tick_params(labelsize="medium")
    if "bert" in name:
        ax.set_xlim(-10,769)
        #ax.set_xlim(0,4)
    elif "gpt2-base-french" in name:
        ax.set_xlim(-10,768)
        #ax.set_xlim(0,4)
    else:
        ax.set_xlim(-10,1280)
        #ax.set_xlim(0,4)
    if case == "er":
        # Base MI line
        ax.axhline(
            y=miplot_basemi.loc[name, f"{putype}_base_mi"], 
            color=palette[mi_renames[f"{putype}_base_mi"]], 
            linestyle="-",
            label= mi_renames[f"{putype}_base_mi"],
        )
        
        y1df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin([
                f'{putype}_I_P_mi', f'{putype}_reconstructed', f'{putype}_encapsulation'
            ]))
        ]
        sns.lineplot(
            data=y1df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax
        )
        
        
        # Title
        namesplit = name.split("_")
        concept, model = namesplit[0], namesplit[1]
        if putype == "gen":
            distname = "Ancestral"
        else:
            distname = "Nucleus"
        ax.set_title(f"{distname}, {concept[0].upper() + concept[1:]},\n {model}", fontsize="medium")
        
        # Y axis
        ax.set_ylabel("Information (Bits)", fontsize="small")#, **ssfont)
        
        # X axis
        #ax.set_xticklabels([])
        #ax.set_xlabel("")
        #ax.set_xlabel("$k$", fontsize=20)#, **ssfont)
        
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        #if i!=0:
            #ax.set_ylabel("")
            #ax.set_yticklabels([])
            #ax.set_yticks([])
        if i!=3:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )       
    ## SECOND ROW
    elif case == "er_perc":
        y2df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin([
                f'{putype}_perc_I_P_mi', f'{putype}_perc_reconstructed', f'{putype}_perc_encapsulation'
            ]))
        ]
        sns.lineplot(
            data=y2df, x="k", y="value", hue="clean_metric", ci="sd",palette=palette, ax=ax
        )
        
        # Base MI line
        #ax.axhline(
        #    y=miplot_basemi.loc[name, f"{putype}_base_mi"], 
        #    color=palette[mi_renames[f"{putype}_base_mi"]], 
        #    linestyle="-",
        #    label= mi_renames[f"{putype}_base_mi"],
        #)
        
        # Title
        #namesplit = name.split("_")
        #concept, model = namesplit[0], namesplit[1]
        #ax.set_title(f"{concept[0].upper() + concept[1:]},\n {model}", fontsize="medium")
        
        # X axis
        #ax.set_xticklabels([])
        #ax.set_xlabel("")

        # Y axis
        ax.set_ylabel("% Total MI", fontsize="small")#, **ssfont)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=None, symbol=None))
        #ax.yaxis.set_tick_params()
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        #if i!=4:
        #    ax.set_ylabel("")
        #    ax.set_yticklabels([])
        #    ax.set_yticks([])
        if i!=7:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )
    ### THIRD ROW          
    elif case == "fth":
        
        fthdfplot = miplot[
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                [f"{putype}_mi_x_Ph_l0", f"{putype}_mi_x_Ph_l1", 
                 f"{putype}_I_P_fth_mi_l0", f"{putype}_I_P_fth_mi_l1"]))
        ]
        sns.lineplot(
            data=fthdfplot, x="k", y="value", hue="clean_metric", 
            ci="sd", palette=palette, ax=ax
        )
        
        ax.set_ylabel("Information (Bits)", fontsize="small")#, **ssfont)
        
        #if i!=8:
        #    ax.set_ylabel("")
        #    ax.set_yticklabels([])
        if i!=11:
            legend = ax.legend().set_visible(False)
        else:
            axhandles, axlabels = ax.get_legend_handles_labels()
            ax.legend(
                axhandles, axlabels, loc='center left', 
                bbox_to_anchor=(1, 0.5), fontsize="small"
            )        
            
    else:
        raise ValueError(f"Incorrect case specification {case}")
        
    

figpath = os.path.join(RESULTS, f"accfthmiplot_ar.png")
fig.savefig(figpath)

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(figpath[:-4]+".pdf") as pp:
    pp.savefig(fig)

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
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]
sample_source = "test"
fig, axes = plt.subplots(2,4,
    gridspec_kw=dict(left=0.08, right=0.99,
                    bottom=0.4, top=0.85),
    dpi=300,
    figsize=(8,4)
)
#fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.05, hspace=0.05)


pairs = [
    #(axes[0,0], "number_gpt2-large", "acc"),
    #(axes[0,1], "number_bert-base-uncased", "acc"),
    #(axes[0,2], "gender_gpt2-base-french", "acc"),
    #(axes[0,3], "gender_camembert-base", "acc"),
    (axes[0,0], "number_gpt2-large", "er_test"),
    (axes[0,1], "number_bert-base-uncased", "er_test"),
    (axes[0,2], "gender_gpt2-base-french", "er_test"),
    (axes[0,3], "gender_camembert-base", "er_test"),
    #(axes[2,0], "number_gpt2-large", "er_gen"),
    #(axes[2,1], "number_bert-base-uncased", "er_gen"),
    #(axes[2,2], "gender_gpt2-base-french", "er_gen"),
    #(axes[2,3], "gender_camembert-base", "er_gen"),
    (axes[1,0], "number_gpt2-large", "fth"),
    (axes[1,1], "number_bert-base-uncased", "fth"),
    (axes[1,2], "gender_gpt2-base-french", "fth"),
    (axes[1,3], "gender_camembert-base", "fth"),
]
for i, (ax, name, case) in enumerate(pairs):
    ax.tick_params(labelsize="medium")
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
        #        'test_reconstructed', 'test_encapsulation', 'test_perc_reconstructed', 
        #        'gen_base_mi', 'gen_I_P_mi',
        #        'gen_reconstructed', 'gen_encapsulation', 'gen_perc_reconstructed']))
        #]
        #sns.lineplot(data=midfplot, x="k", y="value", hue="clean_metric", ci="sd", palette=palette, ax=ax)
        ax2 = ax.twinx()
        y1df = miplot[
            (miplot["k"] != 0) & 
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(test_metrics))
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
        ax.set_ylabel(r"$I(C; \boldsymbol{P}H \mid \boldsymbol{X})", fontsize="medium")#, **ssfont)
        #ax2.set_ylabel("MI/CE Bits", **ssfont)
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles=handles[1:], labels=labels[1:])
        ax.legend().set_title("")
        #ax.legend(fontsize="xx-large")#.fontsize = "x-large"
        ax.legend(fontsize="medium")
        ax2.legend().set_title("")
        ax2.legend(fontsize="medium", loc="upper center")
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
        #        'test_reconstructed', 'test_encapsulation', 'test_perc_reconstructed', 
        #        'gen_base_mi', 'gen_I_P_mi',
        #        'gen_reconstructed', 'gen_encapsulation', 'gen_perc_reconstructed']))
        #]
        #sns.lineplot(data=midfplot, x="k", y="value", hue="clean_metric", ci="sd", palette=palette, ax=ax)
        ax2 = ax.twinx()
        y1df = miplot[
            (miplot["k"] != 0) &
            (miplot["model_concept"] == name) & 
            (miplot["metric"].isin(
                [#'test_base_mi', 'test_I_P_mi','test_reconstructed', 'test_encapsulation',
                'gen_base_mi', 'gen_I_P_mi', 'gen_reconstructed', 'gen_encapsulation',
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
        ax.legend(fontsize="medium")#.fontsize = "x-large"
        ax.set_xlabel("$k$", fontsize="medium")#, **ssfont)
        #ax.set_xticklabels([])
        #ax.set_ylim(-.25,9)
        #ax.yticks(fontsize=20)
        if i!=4:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    else:
        raise ValueError("Incorrect case specification")
        
    
    

figpath = os.path.join(RESULTS, "accfthmiplot.png")
fig.savefig(figpath)

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