#%%
import spacy
import pickle
import sys
#sys.path.append('../../')

import csv
from tqdm import tqdm
#from data.linzen.utils import vinfl

from paths import LINZEN_RAW, LINZEN_PREPROCESSED

#%%
nlp = spacy.load("en_core_web_sm")

#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

#for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.tag_)
#TODO: do the temp thing
data = []
with open(LINZEN_RAW,'r') as f:
    raw = csv.DictReader(f, delimiter='\t')
    for i, record in enumerate(tqdm(raw)):
        orig = record['orig_sentence']
        doc = nlp(orig)
        for token in doc:
            data.append((token.text, token.lemma_, token.pos_, token.tag_))
        

# %%
export_path = f"/cluster/work/cotterell/cguerner/usagebasedprobing/out/tagged_linzen_words.pkl"
    
with open(export_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# %%
#with open(export_path, 'rb') as f:      
#    data = pickle.load(f)
# %%
from datasets import load_dataset
from paths import HF_CACHE

data = load_dataset("wikipedia", "20220301.fr", cache_dir=HF_CACHE)
#load_dataset("wikipedia", "20220301.fr")
data = load_dataset("wikipedia", "20220301.en", cache_dir=HF_CACHE)


# %%
