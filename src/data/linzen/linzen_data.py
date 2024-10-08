#%%
import sys
import os
sys.path.append('../../')
#sys.path.append('..')
#sys.path.append('./src/')

import csv
from tqdm import tqdm
from data.linzen.utils import vinfl

from paths import LINZEN_RAW, DATASETS

def inflect(verb):
    return vinfl[verb]

LINZEN_PREPROCESSED = os.path.join(DATASETS, "preprocessed/linzen_preprocessed.tsv")

#%% 
# Commented out bits inherited from Yoav code
#cases_we_care_about=['1','2','3','4']

data = []
ns = []
with open(LINZEN_RAW,'r') as f:
    raw = csv.DictReader(f, delimiter='\t')
    for i, record in enumerate(tqdm(raw)):
        orig = record['orig_sentence']
        n_i  = record['n_intervening']
        n_di = record['n_diff_intervening']
        ns.append((n_i, n_di))
        vpos = record['verb_pos']
        vindex = int(record['verb_index'])-1
        #if n_i != n_di: continue
        #if n_di in cases_we_care_about:
        sorig = orig.split()
        verb = sorig[vindex]
        iverb = inflect(verb)
        #if verb in ['is','are']: continue # skip because of copular agreement
        sorig[vindex] = "[MASK]"
        masked = " ".join(sorig)
        #print("\t".join([n_di,orig,masked,verb,iverb]))
        data.append([n_di,orig,masked,verb,iverb,vpos,vindex])

# %% Count is/are
#count = 0 
#for sample in data:
#    if sample[3] == "is" or sample[4] == "is":
#        count +=1
    
# %%
with open(LINZEN_PREPROCESSED, "w", newline="") as outfile:
    writer = csv.writer(outfile, delimiter = '\t', lineterminator='\n')
    for record in tqdm(data):
        writer.writerow(record)
# %%
