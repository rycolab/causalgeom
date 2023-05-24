import numpy as np
import pandas as pd

#%% Overall concept marginal
def compute_concept_marginal(ys):
    label_counts = np.unique(ys, return_counts=True)
    assert (label_counts[0] == np.array([0,1])).all(), "Error in label counts"
    p_concept = label_counts[1] / np.sum(label_counts[1])
    return p_concept

#%% Concept pair marginals
def rebuild_concept_pairs(ys, facts, foils):
    pairs_0_1 = []
    for label, fact, foil in zip(y_train, fact_train, foil_train):
        if label == 0:
            pairs_0_1.append((fact, foil, label))
        elif label == 1:
            pairs_0_1.append((foil, fact, label))
        else:
            raise ValueError("Incorrect label")
    df = pd.DataFrame(pairs_0_1, columns = ["c0", "c1", "label"])
    return df

def compute_pair_marginals(ys, facts, foils):
    pairs_df = rebuild_concept_pairs(ys, facts, foils)
    label_counts = pairs_df.groupby(
        ["c0", "c1", "label"])["label"].count().reset_index(name="count")
    label_pivot = label_counts.pivot_table(
        values="count", 
        index=["c0", "c1"],
        columns="label"
    )
    label_pivot.fillna(value=0, inplace=True)
    label_pivot["total"] = label_pivot.sum(axis=1)
    label_pivot["p0"] = label_pivot[0] / label_pivot["total"]
    label_pivot["p1"] = label_pivot[1] / label_pivot["total"]
    pair_probs = label_pivot[["p0", "p1"]].reset_index()
    return pair_probs
