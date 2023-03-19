#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

from tqdm import tqdm
import pickle
import spacy
from datasets import load_dataset

#sys.path.append('../../')

from paths import HF_CACHE, OUT

# %%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "-language", 
        type=str,
        choices=["en", "fr"],
        help="Which language to extract from"
    )
    return argparser.parse_args()

#%%
def process_sample(sample):
    tokens = []
    doc = nlp(sample["text"])
    for token in doc:
        tokens.append(
            dict(
                text=token.text, 
                lemma=token.lemma_, 
                pos=token.pos_,
                tag=token.tag_
            )
        )
    return tokens

def create_temp_files(data):
    temp_nbatches=50
    tempcount = 0
    tagged_tokens = []
    nsamples = len(data)
    for i, obs in enumerate(tqdm(data)):
        tagged_tokens += process_sample(obs)
        if (i + 1) % temp_nbatches == 0 or i == nsamples-1:
            tempfile = os.path.join(
                TEMPDIR, 
                f"temp{tempcount}.pkl"
            )
            with open(tempfile, 'wb') as f:
                pickle.dump(tagged_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            tagged_tokens = []
            tempcount+=1

#%%
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    language = args.language 

    data = load_dataset(
        "wikipedia", f"20220301.{language}", cache_dir=HF_CACHE
    )["train"]
    
    if language == "fr":
        nlp = spacy.load("fr_core_news_sm")
    else: 
        nlp = spacy.load("en_core_web_sm")

    OUT_DIR = os.path.join(OUT, f"wordlist")
    assert os.path.exists(OUT_DIR), f"{OUT_DIR} doesn't exist"

    OUT_FILE = os.path.join(OUT_DIR, f"{language}.pkl")

    TEMPDIR = os.path.join(OUT_DIR, f"temp_{language}")
    os.mkdir(TEMPDIR)

    create_temp_files(data)

#%%
"""

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
"""