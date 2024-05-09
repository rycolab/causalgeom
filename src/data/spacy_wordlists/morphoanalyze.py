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

#sys.path.append('../../')
sys.path.append('./src/')

from utils.dataset_loaders import load_wikipedia
from paths import HF_CACHE, OUT


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
                tag=token.tag_,
                morph=token.morph
            )
        )
    return tokens

def increment_feature_count(feature_dict, token_feature_val):
    feature_count = feature_dict.get(token_feature_val, 0)
    feature_dict[token_feature_val] = feature_count + 1
    return feature_dict

def update_token_dict(token_dict, token_list):
    for token in token_list:
        tt, tlemma, tpos, ttag, tmorph = token["text"].lower(), token["lemma"], token["pos"], token["tag"], str(token["morph"])
        td = token_dict.get(tt, dict(token_lemma={}, token_pos={}, token_tag={}, token_morph={}, count=0))
        td["count"] += 1
        td["token_lemma"] = increment_feature_count(td["token_lemma"], tlemma)
        td["token_pos"] = increment_feature_count(td["token_pos"], tpos)
        td["token_tag"] = increment_feature_count(td["token_tag"], ttag)
        td["token_morph"] = increment_feature_count(td["token_morph"], tmorph)
        token_dict[tt] = td
    return token_dict

"""
def merge_token_dicts(dict1, dict2):
    #take two token dicts and combine them

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
"""

def export_token_dict(token_dict, outfile, switch=None):
    if switch is not None:
        switch_outfile = f"{outfile[:-4]}_{switch}.pkl"
    else:
        switch_outfile = outfile
    with open(switch_outfile, 'wb') as f:
        pickle.dump(token_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Token dict exported to: {switch_outfile}")


def create_token_dict(data, outfile, backup_batches):
    nsamples = len(data)
    switch=0
    token_dict = {}
    for i, obs in enumerate(tqdm(data)):
        token_dict = update_token_dict(token_dict, process_sample(obs))
        if (i + 1) % backup_batches == 0 or i == nsamples-1:
            export_token_dict(token_dict, outfile, switch)
            if switch == 0:
                switch = 1
            else:
                switch = 0
    return token_dict
    

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Process hidden states')
    argparser.add_argument(
        "-language", 
        type=str,
        choices=["en", "fr"],
        help="Which language to extract from"
    )
    argparser.add_argument(
        "-backup_batches", 
        type=int,
        default=100,
        help="Number of batches before temp export"
    )
    return argparser.parse_args()
    
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    language = args.language 
    backup_batches = args.backup_batches 
    #language = "fr"
    data = load_wikipedia(language)
    data.shuffle()
    
    if language == "fr":
        nlp = spacy.load("fr_core_news_sm")
    elif language == "en": 
        nlp = spacy.load("en_core_web_sm")
    else:
        raise ValueError(f"Unsupported language: {language}")

    OUT_DIR = os.path.join(DATASETS, f"{language}")
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.info(f"Created output directory: {OUT_DIR}")
    #assert os.path.exists(OUT_DIR), f"{OUT_DIR} doesn't exist"

    OUT_FILE = os.path.join(OUT_DIR, f"{language}_wiki_wordlist.pkl")

    #TEMPDIR = os.path.join(OUT_DIR, f"temp_{language}")
    #os.mkdir(TEMPDIR)

    #create_temp_files(data)
    token_dict = create_token_dict(data, OUT_FILE, backup_batches)
    export_token_dict(token_dict, OUT_FILE)
    