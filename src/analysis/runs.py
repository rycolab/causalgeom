#%%
import json
import pandas as pd
import os
import pickle

jsondir = "/cluster/work/cotterell/cguerner/usagebasedprobing/out/real_runs"
jsonlist = os.listdir(jsondir)
jsondictlist = []

for json in jsonlist:
    if not json.endswith(".pkl"):
        continue
    with open(os.path.join(jsondir, json), "rb") as f:
        data = pickle.load(f)
        base = {
            "run": data["run"],
            "maj_acc_train": data["maj_acc_train"],
            "maj_acc_val": data["maj_acc_val"],
            "maj_acc_test": data["maj_acc_test"]
        }
        base = base | data["run_args"]
        inlp_dict = data["inlp"]
        inlp_dict["method"] = "inlp"
        diag_rlace_dict = data["diag_rlace"]
        diag_rlace_dict["method"] = "diag_rlace"
        functional_rlace_dict = data["functional_rlace"]
        functional_rlace_dict["method"] = "functional_rlace"
        jsondict = {
            "base": base,
            "inlp": inlp_dict,
            "diag_rlace": diag_rlace_dict,
            "functional_rlace": functional_rlace_dict
        }
        jsondictlist.append(jsondict)

data = pd.DataFrame.from_records(jsondictlist)
base = pd.json_normalize(data["base"])
inlp = pd.json_normalize(data["inlp"])
diag_rlace = pd.json_normalize(data["diag_rlace"])
func_rlace = pd.json_normalize(data["functional_rlace"])

data = pd.DataFrame()
for i in range(base.shape[0]):
    baserow = base.loc[i,:]
    inlprow = inlp.loc[i,:]
    diag_rlace_row = diag_rlace.loc[i,:]
    func_rlace_row = func_rlace.loc[i,:]
    inlp_full = pd.concat([baserow, inlprow], axis=0)
    diag_rlace_full = pd.concat([baserow, diag_rlace_row], axis=0)
    func_rlace_full = pd.concat([baserow, func_rlace_row], axis=0)
    newrows = pd.concat([inlp_full, diag_rlace_full, func_rlace_full], axis=1).T
    data = pd.concat([data, newrows],axis=0)

#%%
#data.dropna(
#    subset = ["method", "rank","rlace_niter", "run","optim_best_acc","diag_acc_original",
#              "diag_acc_projected_test","diag_acc_projected_train",
#              "diag_acc_comp_projected_train","diag_acc_comp_projected_test",
#              "lm_acc_original","lm_acc_projected_test","lm_acc_projected_train",
#              "lm_acc_comp_projected_train","lm_acc_comp_projected_test"]
#)
acc_res = data.groupby(["rank","method", "rlace_niter"]).agg(
    {
        "run": "count", 
        "maj_acc_test": "mean",
        "optim_best_acc":"mean", 
        "diag_acc_original":"mean", 
        "diag_acc_projected_test":"mean",
        "diag_acc_projected_train":"mean",
        "diag_acc_comp_projected_train":"mean",
        "diag_acc_comp_projected_test":"mean",
        "lm_acc_original":"mean",
        "lm_acc_projected_test":"mean",
        "lm_acc_projected_train":"mean",
        "lm_acc_comp_projected_train":"mean",
        "lm_acc_comp_projected_test":"mean"
    }
)
acc_res.to_csv("../../out/results_acc.csv")

data["optim_best_loss"].fillna(value=0, inplace=True)
loss_res = data.groupby(["rank","method", "rlace_niter"]).agg(
    {
        "run": "count", 
        "maj_acc_test": "mean",
        "optim_best_loss":"mean", 
        "diag_loss_original":"mean", 
        "diag_loss_projected_test":"mean",
        "diag_loss_projected_train":"mean",
        "diag_loss_comp_projected_train":"mean",
        "diag_loss_comp_projected_test":"mean",
        "lm_loss_original":"mean",
        "lm_loss_projected_test":"mean",
        "lm_loss_projected_train":"mean",
        "lm_loss_comp_projected_train":"mean",
        "lm_loss_comp_projected_test":"mean"
    }
)
loss_res.to_csv("../../out/results_loss.csv")
# %%
