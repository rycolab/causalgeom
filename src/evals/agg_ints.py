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
#import torch
import random 
#from scipy.special import softmax
#from scipy.stats import entropy
from tqdm import trange

sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
#from evals.kl_eval import load_run_output
#from utils.dataset_loaders import load_processed_data

#from evals.usage_eval import diag_eval, usage_eval
#from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST, get_concept_name
#from models.fit_kde import load_data
#from data.embed_wordlists.embedder import load_concept_token_lists
#from evals.kl_eval import load_run_Ps, load_run_output, \
#    compute_eval_filtered_hs, load_model_eval, compute_kl, \
#        renormalize, get_distribs

#from analysis.format_res import get_best_runs
#from data.filter_generations import load_filtered_hs, load_filtered_hs_wff

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


# %%
int_path = os.path.join(RESULTS, "int")
csvs = os.listdir(int_path)

# %%
run_ints = []
for csvname in csvs:
    #csvname = csvs[0]
    run_int_path = os.path.join(int_path, csvname)
    run_int = pd.read_csv(run_int_path, index_col=0)
    run_ints.append(run_int.to_dict()["0"])

# %%
df = pd.DataFrame.from_records(run_ints)
#df.to_csv(os.path.join(RESULTS, "intres.csv"), index=False)

#%%
df["model_concept"] = df["concept"] + "_" + df["model"]
#df = df[[
#    'model_concept', "run",  #'dev_total_samples', 'dev_nsamples',#'test_total_samples', 'test_nsamples',
#    'base_correct', 'base_correct_highest', 'base_correct_highest_concept',
#    'I_P_correct', 'I_P_l0_highest', 'I_P_l1_highest',
#    'I_P_l0_highest_concept', 'I_P_l1_highest_concept',
#    'avgh_inj0_correct', 'avgh_inj0_l0_highest',
#    'avgh_inj0_l0_highest_concept', 'avgh_inj1_correct',
#    'avgh_inj1_l1_highest', 'avgh_inj1_l1_highest_concept',
#    'avgp_inj0_correct', 'avgp_inj0_l0_highest',
#    'avgp_inj0_l0_highest_concept', 'avgp_inj1_correct',
#    'avgp_inj1_l1_highest', 'avgp_inj1_l1_highest_concept']]

df = df[[
    'model_concept', "run",  #'dev_total_samples', 'dev_nsamples',#'test_total_samples', 'test_nsamples',
    'base_correct', 'base_correct_highest_concept',
    'I_P_correct', 'I_P_l0_highest_concept', 'I_P_l1_highest_concept',
    'avgh_inj0_correct', 'avgh_inj0_l0_highest_concept', 
    'avgh_inj1_correct', 'avgh_inj1_l1_highest_concept',
    'avgp_inj0_correct', 'avgp_inj0_l0_highest_concept', 
    'avgp_inj1_correct', 'avgp_inj1_l1_highest_concept']]

numlist = ['base_correct', 'base_correct_highest_concept',
    'I_P_correct', 'I_P_l0_highest_concept', 'I_P_l1_highest_concept',
    'avgh_inj0_correct', 'avgh_inj0_l0_highest_concept', 
    'avgh_inj1_correct', 'avgh_inj1_l1_highest_concept',
    'avgp_inj0_correct', 'avgp_inj0_l0_highest_concept', 
    'avgp_inj1_correct', 'avgp_inj1_l1_highest_concept']
for col in numlist:
    df[col] = df[col].astype(float)

meandf = df.groupby(["model_concept"]).mean()

df.to_csv(os.path.join(RESULTS, "int_full.csv"))
meandf.to_csv(os.path.join(RESULTS, "int.csv"))

#%%
longdf = pd.melt(df, id_vars=["model_concept", "run"], value_vars=numlist)

cleanmetricnames = {
    'base_correct': "Orig. Accuracy",
    'base_correct_highest_concept': "Orig. Top Concept",
    'I_P_correct': "Ph Accuracy", 
    'I_P_l0_highest_concept': "Ph C=0 Top Concept",
    'I_P_l1_highest_concept': "Ph C=1 Top Concept",
    'avgh_inj0_correct': "Do C=0 Accuracy",
    'avgh_inj0_l0_highest': "Do C=0 Top",
    'avgh_inj0_l0_highest_concept': "Do C=0 Top Concept",
    'avgh_inj1_correct': "Do C=1 Accuracy",
    'avgh_inj1_l1_highest': "Do C=1 Top",
    'avgh_inj1_l1_highest_concept': "Do C=1 Top Concept",
    'avgp_inj0_correct': "Do C=0 Accuracy",
    'avgp_inj0_l0_highest': "Do C=0 Top",
    'avgp_inj0_l0_highest_concept': "Do C=0 Top Concept",
    'avgp_inj1_correct': "Do C=1 Accuracy",
    'avgp_inj1_l1_highest': "Do C=1 Top",
    'avgp_inj1_l1_highest_concept': "Do C=1 Top Concept",
}
longdf["clean_variable"] = [cleanmetricnames[x] for x in longdf["variable"]]
#longdf["type"] = longdf["variable"].apply(lambda x: x.split("_")[0])
#longdf["rest"] = longdf["variable"].apply(lambda x: "_".join(x.split("_")[1:]))


#%%
avgh_cols = [
    'base_correct', 'base_correct_highest', 'base_correct_highest_concept',
    'I_P_correct', 'I_P_l0_highest_concept', 'I_P_l1_highest_concept',
    'avgh_inj0_correct', 'avgh_inj0_l0_highest',
    'avgh_inj0_l0_highest_concept', 'avgh_inj1_correct',
    'avgh_inj1_l1_highest', 'avgh_inj1_l1_highest_concept'
]
longdf_avgh = longdf[longdf["variable"].isin(avgh_cols)]
avgp_cols = [
    'base_correct', 'base_correct_highest', 'base_correct_highest_concept',
    'I_P_correct', 'I_P_l0_highest_concept', 'I_P_l1_highest_concept',
    'avgp_inj0_correct', 'avgp_inj0_l0_highest',
    'avgp_inj0_l0_highest_concept', 'avgp_inj1_correct',
    'avgp_inj1_l1_highest', 'avgp_inj1_l1_highest_concept'
]
longdf_avgp = longdf[longdf["variable"].isin(avgp_cols)]

# %%
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

intervention = "avgp"

clean_names_list = list(set(cleanmetricnames.values()))
cols = sns.color_palette("bright", len(clean_names_list))
palette = {}
for k, v in zip(clean_names_list, cols):
    palette[k] = v

fig, axes = plt.subplots(1,4,
    gridspec_kw=dict(left=0.08, right=0.99,
                    bottom=0.4, top=0.85),
    figsize=(8,3),
    dpi=300
    )
#fig.set_size_inches(8, 3,forward=True)
#fig.set_dpi(100)
fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

pairs = [
    (axes[0], "number_gpt2-large", "acc"),
    (axes[1], "number_bert-base-uncased", "acc"),
    (axes[2], "gender_gpt2-base-french", "acc"),
    (axes[3], "gender_camembert-base", "acc"),
]
for i, (ax, name, case) in enumerate(pairs):
    ax.tick_params(labelsize="medium")

    if intervention == "avgp":
        plotdf = longdf_avgp[longdf_avgp["model_concept"] == name]
    elif intervention == "avgh":
        plotdf = longdf_avgh[longdf_avgh["model_concept"] == name]
    else:
        raise ValueError("Incorrect intervention type")
    sns.barplot(
        data=plotdf, y="value", x="clean_variable", ci = "sd", palette=palette, 
        ax=ax
    )
    #ax.axhline(y=accbase.loc[name, "maj_acc_test"], color="green", linestyle="dashed", label="Majority Accuracy")
    #ax.axhline(y=accbase.loc[name, "test_accuracy"], color="r", linestyle="-")
    #ax.set_ylim(.5, 1)
    #ax.set_ylabel("Accuracy")
    
    namesplit = name.split("_")
    concept, model = namesplit[0], namesplit[1]
    ax.set_title(f"{concept[0].upper() + concept[1:]},\n {model}", fontsize="medium")

    #removing legend titles
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles=handles[1:], labels=labels[1:])
    #ax.legend().set_title("")
    #ax.get_shared_x_axes().join(ax, axes[1,i])
    #ax.set_xticklabels([])
    ax.set_xlabel("")
    xlabels = ax.get_xticklabels()
    for x in xlabels:
        if "number" in name:
            #x.set_text(x.get_text().replace("C=0", "$C = \textsf{sg}$").replace("C=1","$C = \textsf{pl}$"))
            x.set_text(x.get_text().replace("C=0", "C=sg").replace("C=1","C=pl"))
        elif "gender" in name:
            #x.set_text(x.get_text().replace("C=0", "$C = \textsf{masc}$").replace("C=1","$C = \textsf{fem}$"))
            x.set_text(x.get_text().replace("C=0", "C=masc").replace("C=1","C=fem"))
        else:
            raise ValueError("Wrong name")
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize="x-small")
    #ax.legend(fontsize="xx-large")
    ax.set_ylabel("Accuracy", fontsize="medium")

    if i!=0:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    #, **ssfont)


from matplotlib.backends.backend_pdf import PdfPages
#fig.subplots_adjust(bottom=0.5)
figpath = os.path.join(RESULTS, f"controlled_{intervention}.pdf")
with PdfPages(figpath) as pp:
    pp.savefig(fig)
fig.savefig(figpath[:-4]+".png")

# %%
