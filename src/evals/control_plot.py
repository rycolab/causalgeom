import warnings
import logging
import os
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import pickle

warnings.filterwarnings("ignore")

#ssfont = {'fontname':'Times New Roman'}
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

from google.colab import files


uploaded = files.upload()
#TODO: fix this to fetch correct files in results

#res = pd.read_csv(io.BytesIO(uploaded['finaleval_plotdf.csv']), index_col=0)
df = pd.read_csv('fig3_intres_input.csv')
df

metric_orders = {
    'base_correct': 0,
    'erased_correct': 1,
    'corr_erased_correct': 2,
    'do_correct': 3
}
df["metric_order"] = df["metric"].apply(lambda x: metric_orders[x])
df.sort_values(by=["concept", "model_name", "run_path", "iteration", "metric_order", "metric"], inplace=True)
df

#with open("int_maj_accs.pkl", "rb") as f:
#    maj_accs = pickle.load(f)

#maj_accs
maj_accs_df = pd.read_csv('int_maj_accs.csv', index_col=0)
maj_accs = maj_accs_df.to_dict()["maj_acc"]
maj_accs

cleanmetricnames = {
    'base_correct': "Orig. Acc.",
    'erased_correct': "Erased Acc.",
    'corr_erased_correct': "Corr. Erased Acc.",
    'do_correct': "Do Acc."
    #'base_correct_l0': "Orig. Acc. (0 fact)",
    #'base_correct_l1': "Orig. Acc. (1 fact)",
    #'base_correct_highest_concept': "Orig. Top Concept",
    #'I_P_correct_l0': "Erased Acc. (0 fact)",
    #'I_P_correct_l1': "Erased Acc. (1 fact)",
    #'I_P_l0_highest_concept': "Ph C=0 Top Concept",
    #'I_P_l1_highest_concept': "Ph C=1 Top Concept",
    #'avgh_inj0_correct': "Do C=0 Acc",
    #'avgh_inj0_l0_highest': "Do C=0 Top",
    #'avgh_inj0_l0_highest_concept': "Do C=0 Top Concept",
    #'avgh_inj1_correct': "Do C=1 Acc",
    #'avgh_inj1_l1_highest': "Do C=1 Top",
    #'avgh_inj1_l1_highest_concept': "Do C=1 Top Concept",
    #'avgp_inj0_correct': "Success Rate Do C=0",
    #'avgp_inj0_l0_highest': "Do C=0 Top",
    #'avgp_inj0_l0_highest_concept': "Do C=0 Top Concept",
    #'avgp_inj1_correct': "Success Rate Do C=1",
    #'avgp_inj1_l1_highest': "Do C=1 Top",
    #'avgp_inj1_l1_highest_concept': "Do C=1 Top Concept",
}

df["clean_metric"] = [cleanmetricnames[x] for x in df["metric"]]

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


try:
    fig.clf()
except NameError:
    logging.info("First figure creation")

#intervention = "avgp"

#clean_names_list = list(set(cleanmetricnames.values()))
#cols = sns.color_palette("bright", len(clean_names_list))
#palette = {}
#for k, v in zip(clean_names_list, cols):
#    palette[k] = v

palette = {
    'Orig. Acc.': (0.00784313725490196, 0.24313725490196078, 1.0),
    'Erased Acc.': (0.9098039215686274, 0.0, 0.043137254901960784),
    'Do Acc.':  (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
    #(0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
    #'Erased Acc. (1 fact)': (0.0, 0.8431372549019608, 1.0),
    #'Success Rate Do C=0': (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    #'Success Rate Do C=1': (1.0, 0.48627450980392156, 0.0),
  }

fig, axes = plt.subplots(1,3,
    gridspec_kw=dict(left=0.15, right=0.95,
                    bottom=0.22, top=0.85),
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
    (axes[0], "number", "gpt2-large", "anc"),
    (axes[1], "number", "llama2", "anc"),
    (axes[2], "gender", "gpt2-base-french", "anc"),
    #(axes[1], "number_gpt2-large", "acc", "nuc"),
    #(axes[3], "gender_gpt2-base-french", "acc", "nuc"),
]
for i, (ax, concept, model, putype) in enumerate(pairs):
    ax.tick_params(labelsize="medium")

    # Nucleus handling
    if putype == "nuc":
        nucleus=True
    elif putype == "anc":
        nucleus=False
    else:
        raise ValueError("Incorrect putype")

    if concept.startswith("number"):
        maj_acc = maj_accs["number"]
    elif concept.startswith("gender"):
        maj_acc = maj_accs["gender"]
    else:
        raise ValueError("Incorrect name")

    plotdf = df[
        (df["concept"] == concept) &
        (df["model_name"] == model) &
        (df["metric"].isin([
            'base_correct', 'erased_correct', 'do_correct']))
    ]
    sns.barplot(
        data=plotdf, y="value", x="clean_metric", ci = "sd", palette=palette,
        ax=ax
    )

    ax.axhline(y=maj_acc, color="red", linestyle="dashed", label="Majority")

    #ax.set_ylabel("Accuracy")

    #namesplit = name.split("_")
    #concept, model = namesplit[0], namesplit[1]
    #if putype == "anc":
    #    distname = "Ancestral"
    #elif putype == "nuc":
    #    distname = "Nucleus"
    #else:
    #    distname = ""
    model = model.replace("gpt", "GPT")
    model = model.replace("llama2", "Llama2")
    ax.set_title(f"{model},\n {concept[0].upper() + concept[1:]}", fontsize="medium")
    #ax.set_title(f"{concept[0].upper() + concept[1:]}", fontsize="medium")


    ax.set_xlabel("")
    xlabels = ax.get_xticklabels()
    #for x in xlabels:
    #    if "number" in name:
    #        #x.set_text(x.get_text().replace("C=0", "$C = \textsf{sg}$").replace("C=1","$C = \textsf{pl}$"))
    #        x.set_text(x.get_text().replace("0", "sg").replace("1","pl"))
    #    elif "gender" in name:
    #        #x.set_text(x.get_text().replace("C=0", "$C = \textsf{masc}$").replace("C=1","$C = \textsf{fem}$"))
    #        x.set_text(x.get_text().replace("0", "msc").replace("1","fem"))
    #    else:
    #        raise ValueError("Wrong name")
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize="x-small")

    if i == 0:
        ax.set_ylabel("Accuracy", fontsize="medium")
    else:
        ax.set_ylabel("")
    #if i!=0:
    #
    #    ax.set_yticklabels([])

    if i == 2:
        ax.legend(fontsize="small")
    #, **ssfont)

plt.tight_layout(pad=0)
plt.show()


from matplotlib.backends.backend_pdf import PdfPages
#fig.subplots_adjust(bottom=0.5)
figpath =  "controlled_avgp.pdf"
with PdfPages(figpath) as pp:
    pp.savefig(fig)
fig.savefig(figpath[:-4]+".png")