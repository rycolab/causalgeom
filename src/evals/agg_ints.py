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



#%%
def process_int_df(sample_df, nucleus):
    sample_df["nucleus"] = nucleus
    key_columns = ['model','concept', "nucleus", 'run']
    split_key_columns = key_columns + ['case']
    split_metric_cols = ["base_correct", "I_P_correct"]
    mean_metric_cols = [
        "case",
        "base_correct", "I_P_correct",
        "base_correct_highest_concept", 
        'avgp_inj0_correct', 'avgp_inj0_l0_highest_concept', 
        'avgp_inj1_correct', 'avgp_inj1_l1_highest_concept'
    ]
    split_metrics = sample_df[
        split_key_columns + split_metric_cols
    ].groupby(split_key_columns).mean().reset_index()
    longsplitmetrics = pd.melt(
        split_metrics, 
        id_vars=split_key_columns, 
        var_name="long_metric"
    )
    longsplitmetrics["metric"] = (
        longsplitmetrics["long_metric"] + 
        longsplitmetrics["case"].apply(lambda x: f"_l{x}") 
    )
    longsplitmetrics.drop(["long_metric", "case"], axis=1, inplace=True)

    mean_metrics = sample_df.groupby(key_columns).mean()[mean_metric_cols].reset_index()
    longmeanmetrics = pd.melt(
        mean_metrics, 
        id_vars=key_columns, 
        var_name="metric"
    )
    all_metrics = pd.concat([longsplitmetrics, longmeanmetrics], axis=0)
    order_dict = {
        'case': 0, 'base_correct': 1, 
        'base_correct_l0': 2, 'base_correct_l1': 3, 
        'base_correct_highest_concept': 4,
        'I_P_correct': 5, 'I_P_correct_l0': 6,'I_P_correct_l1': 7, 
        'avgp_inj0_correct': 8, 'avgp_inj0_l0_highest_concept': 9, 
        'avgp_inj1_correct': 10, 'avgp_inj1_l1_highest_concept':11}
    all_metrics["order"] = [order_dict[x] for x in all_metrics["metric"]]
    all_metrics.sort_values(by = "order", inplace=True, ascending=True)
    return all_metrics


# %%
int_path = os.path.join(RESULTS, "int_samples")
csvs = os.listdir(int_path)

# %%
run_ints = []
for csvname in csvs:
    #csvname = csvs[0]
    run_int_path = os.path.join(int_path, csvname)
    run_samples = pd.read_csv(run_int_path, index_col=0)
    #TODO: fix this
    nucleus = "True" in run_int_path
    int_metrics = process_int_df(run_samples, nucleus)
    run_ints.append(int_metrics)

df = pd.concat(run_ints, axis=0).reset_index(drop=True)
df.sort_values(
    by = ["model", "concept", "nucleus", "run", "order"], 
    inplace=True
)

#%%
maj_acc = df[df["metric"] == "case"].groupby(["concept"])["value"].mean()
maj_acc_gender = maj_acc.loc["gender"]
maj_acc_number = 1 - maj_acc.loc["number"]
maj_accs = {
    "number": maj_acc_number,
    "gender": maj_acc_gender,
}
maj_acc_outfile = os.path.join(RESULTS, "majaccs.pkl")
with open(maj_acc_outfile, "wb") as f:
    pickle.dump(maj_accs, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%
cleanmetricnames = {
    'base_correct': "Orig. Accuracy",
    'base_correct_l0': "Orig. Accuracy C=0",
    'base_correct_l1': "Orig. Accuracy C=1",
    'base_correct_highest_concept': "Orig. Top Concept",
    'I_P_correct': "Erased Accuracy", 
    'I_P_correct_l0': "Erased Accuracy C=0",
    'I_P_correct_l1': "Erased Accuracy C=1",
    #'I_P_l0_highest_concept': "Ph C=0 Top Concept",
    #'I_P_l1_highest_concept': "Ph C=1 Top Concept",
    #'avgh_inj0_correct': "Do C=0 Accuracy",
    #'avgh_inj0_l0_highest': "Do C=0 Top",
    #'avgh_inj0_l0_highest_concept': "Do C=0 Top Concept",
    #'avgh_inj1_correct': "Do C=1 Accuracy",
    #'avgh_inj1_l1_highest': "Do C=1 Top",
    #'avgh_inj1_l1_highest_concept': "Do C=1 Top Concept",
    'avgp_inj0_correct': "Do C=0 Accuracy",
    'avgp_inj0_l0_highest': "Do C=0 Top",
    'avgp_inj0_l0_highest_concept': "Do C=0 Top Concept",
    'avgp_inj1_correct': "Do C=1 Accuracy",
    'avgp_inj1_l1_highest': "Do C=1 Top",
    'avgp_inj1_l1_highest_concept': "Do C=1 Top Concept",
}
df.drop(df[df["metric"] == "case"].index, axis=0, inplace=True)
df["clean_metric"] = [cleanmetricnames[x] for x in df["metric"]]

df["model_concept"] = df["concept"] + "_" + df["model"]
df.drop(["model", "concept", "order", "run"], axis=1, inplace=True)
df.to_csv(os.path.join(RESULTS, "intres.csv"), index=False)

#%%
"""
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
#longdf = pd.melt(df, id_vars=["model_concept", "run"], value_vars=numlist)
"""
#%%

#longdf["type"] = longdf["variable"].apply(lambda x: x.split("_")[0])
#longdf["rest"] = longdf["variable"].apply(lambda x: "_".join(x.split("_")[1:]))

# %%
import seaborn as sns
import matplotlib.pyplot as plt 
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"


try:
    fig.clf()
except NameError:
    logging.info("First figure creation")

#intervention = "avgp"

clean_names_list = list(set(cleanmetricnames.values()))
cols = sns.color_palette("bright", len(clean_names_list))
palette = {}
for k, v in zip(clean_names_list, cols):
    palette[k] = v

fig, axes = plt.subplots(1,2,
    gridspec_kw=dict(left=0.16, right=0.99,
                    bottom=0.36, top=0.85),
    sharey="row",
    #sharex="col",
    figsize=(4,3),
    dpi=300
)
#fig.set_size_inches(8, 3,forward=True)
#fig.set_dpi(100)
fig.tight_layout()
#fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

pairs = [
    (axes[0], "number_gpt2-large", "acc", "anc"),
    (axes[1], "gender_gpt2-base-french", "acc", "anc"),
    #(axes[1], "number_gpt2-large", "acc", "nuc"),
    #(axes[3], "gender_gpt2-base-french", "acc", "nuc"),
]
for i, (ax, name, case, putype) in enumerate(pairs):
    ax.tick_params(labelsize="medium")
    
    # Nucleus handling
    if putype == "nuc":
        nucleus=True
    elif putype == "anc":
        nucleus=False
    else:
        raise ValueError("Incorrect putype")

    if name.startswith("number"):
        maj_acc = maj_accs["number"]
    elif name.startswith("gender"):
        maj_acc = maj_accs["gender"]
    else:
        raise ValueError("Incorrect name")
        
    plotdf = df[
        (df["model_concept"] == name) &
        (df["nucleus"] == nucleus) & 
        (df["metric"].isin([
            "base_correct", "base_correct_highest_concept", 
            "I_P_correct_l0","I_P_correct_l1",  
            'avgp_inj0_correct', 'avgp_inj0_l0_highest_concept', 
            'avgp_inj1_correct', 'avgp_inj1_l1_highest_concept']))
    ]
    sns.barplot(
        data=plotdf, y="value", x="clean_metric", ci = "sd", palette=palette, 
        ax=ax
    )
    
    ax.axhline(y=maj_acc, color="red", linestyle="dashed", label="Majority")
    
    #ax.set_ylabel("Accuracy")
    
    namesplit = name.split("_")
    concept, model = namesplit[0], namesplit[1]
    #if putype == "anc":
    #    distname = "Ancestral"
    #elif putype == "nuc":
    #    distname = "Nucleus"
    #else:
    #    distname = ""
    model = model.replace("gpt", "GPT")
    ax.set_title(f"{model},\n {concept[0].upper() + concept[1:]}", fontsize="medium")
    #ax.set_title(f"{concept[0].upper() + concept[1:]}", fontsize="medium")


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
    
    if i == 0:
        ax.set_ylabel("Accuracy", fontsize="medium")
    else:
        ax.set_ylabel("")
    #if i!=0:
    #    
    #    ax.set_yticklabels([])
    
    if i == 1:
        ax.legend(fontsize="small")
    #, **ssfont)

plt.tight_layout(pad=0)
plt.show()


from matplotlib.backends.backend_pdf import PdfPages
#fig.subplots_adjust(bottom=0.5)
figpath = os.path.join(RESULTS, f"controlled_avgp.pdf")
with PdfPages(figpath) as pp:
    pp.savefig(fig)
fig.savefig(figpath[:-4]+".png")

# %%
