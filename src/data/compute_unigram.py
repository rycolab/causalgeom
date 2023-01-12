#%%
#TODO: REFACTOR FOR USE BY OTHERS

import numpy as np
from scipy.stats import entropy

INFILE_FOLDER = "/cluster/scratch/cguerner/thesis_data/unigram_freqs/"
OUTFILE_PATH = "data/unigram_distribs/multibert.npy"

#%%
dataset = "bookcorpus" # "wikipedia", "bookcorpus" 
vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
filedict = {}
for val in vals:
    strval = str(val).replace(".", "_")
    filepath = os.path.join(INFILE_FOLDER, f"{dataset}_{strval}_percent.npy")
    ids, counts = np.load(filepath)
    filedict[strval] = counts

#%% Helpers
def get_log_unigram(counts, smoothed=True):
    if smoothed:
        counts = counts + 1
    probs = counts / np.sum(counts)
    return np.log(probs)

def get_js_divergence(pdist, qdist):
    mdist = 0.5 * (pdist + qdist)
    js = 0.5 * (entropy(pdist, mdist) + entropy(qdist, mdist))
    return js

#%% Compare sample counts
pdist = get_log_unigram(filedict["0_3"])

all_js = []
for size, counts in filedict.items():
    if size == "0_3":
        continue
    else:
        q = get_log_unigram(counts)
        js = get_js_divergence(pdist, q)
        all_js.append((size, js))

all_js 

#%% COMPUTE COMBINED LOG UNIGRAM
wiki_token_ids, wiki_counts = np.load("/cluster/scratch/cguerner/thesis_data/unigram_freqs/wikipedia_0_2_percent.npy")
bc_token_ids, bc_counts = np.load("/cluster/scratch/cguerner/thesis_data/unigram_freqs/bookcorpus_0_2_percent.npy")

combined_counts = wiki_counts + bc_counts
probs = get_log_unigram(combined_counts)

np.save(OUTFILE_PATH, np.stack((wiki_token_ids, probs)))
