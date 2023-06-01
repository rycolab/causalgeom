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
import torch

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT


from utils.lm_loaders import get_tokenizer, get_V, get_model
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%%#################
# Generating Text  #
####################
def check_eos(latest_tokens, eos_token=50256):
    if tokens[:,-1].unique().shape[0] == 1 and tokens[:,-1].unique()[0] == eos_token:
        return False
    else:
        return True

def adjust_past_key_values(past_key_values, max_len = 1024):
    new_past_key_values = ()
    for i in range(len(past_key_values)):
        new_sub_values = ()
        for j in range(len(past_key_values[i])):
            vals = past_key_values[i][j]
            assert vals.shape[2]==max_len, "Incorrect shape past_key_values"
            new_sub_values += (vals[:,:,-1023:,:],)
        new_past_key_values += (new_sub_values,)
    return new_past_key_values
    
def generate_sequence_until_eos(model, prompt_ids, batch_size, seed, device="cpu", max_length=1024, eos_token=50256, P=None):
    torch.manual_seed(seed)
    tokens = prompt_ids.repeat(batch_size, 1)
    token_h_pairs = []
    sm = torch.nn.Softmax(1)#.to(device)
    past_key_values = None
    counter = 0
    all_tokens = [tokens]
    while (check_eos(tokens[:,-1], eos_token) or (counter==0)):
        if past_key_values is not None and past_key_values[0][0].shape[2] == max_length:
            #copy = past_key_values[0][0].clone().detach()
            past_key_values = adjust_past_key_values(past_key_values, max_length)
        #if tokens.shape[1] > 1024:
        #    tokens = tokens[:,-1024:]
        #else:
        #    input_tokens = tokens
        with torch.no_grad():
            output = model(
                input_ids=tokens,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
        #final_logits = output.logits[-1]#.cpu()
        past_key_values = output.past_key_values
        hs = output.hidden_states[-1][:,-1,:]#.squeeze().squeeze()#.cpu()
        if P is not None:
            hs = torch.matmul(P, hs.T).T
        check_logits = model.lm_head(hs)
        #assert (torch.isclose(final_logits[-1,:], check_logits, atol = 1e-05)).all().item(), "Logits not equal"
        probs = sm(check_logits)
        sampled_tokens = torch.multinomial(probs, 1)
        save_h_indicator = []
        for i in range(batch_size):
            if tokens[i, -1] == eos_token and counter > 0:
                sampled_tokens[i] = eos_token
                save_h_indicator.append(False)
            else:
                save_h_indicator.append(True)
                #torch.cat((zip_tokens, zip_sampled_token.unsqueeze(0)), dim=-1)
        #tokens = torch.cat((tokens, sampled_tokens), dim=-1)
        all_tokens.append(sampled_tokens)
        tokens = sampled_tokens
        for h, token, save_h in zip(hs.cpu(), sampled_tokens, save_h_indicator):
            if save_h:
                token_h_pairs.append((h, token.item()))
        counter+=1
        #logging.info(f"Number of tokens so far: {counter}") 
    all_tokens = torch.hstack(all_tokens).cpu()
    return all_tokens, token_h_pairs

#%%
def generate_multiple_sequences(model, prompt_ids, batch_size, seed, export_index, 
    outdir, n_generations=1, device="cpu", P=None):
    all_token_h_pairs = []
    all_lengths = []
    for i in range(n_generations):
        _, token_h_pairs = generate_sequence_until_eos(model, prompt_ids, device)
        all_lengths.append(len(token_h_pairs))
        all_token_h_pairs += token_h_pairs
    
    export_path = os.path.join(outdir, f"generations_seed_{seed}_{export_index}.pkl")
    with open(export_path, 'wb') as f:
        pickle.dump(all_token_h_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.cuda.empty_cache()

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Generate from model')
    argparser.add_argument(
        "-model",
        type=str,
        choices=GPT2_LIST,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-P",
        type=bool,
        choices=[True, False],
        help="Whether to load and apply a P for this set of generations",
        dest="use_P"
    )
    argparser.add_argument(
        "-export_index",
        type=int,
        help="Special index for this job",
    )
    return argparser.parse_args()

#%%
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    #model_name = args.model
    #use_P = args.use_P
    #export_index = args.export_index
    batch_size = 3
    model_name = "gpt2-large"
    use_P = True
    export_index = 0


    device = get_device()
    #device = "cpu"
    model = get_model(model_name).to(device)
    tokenizer = get_tokenizer(model_name)

    prompt_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)

    if model_name == "gpt2-large":
        run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")
    else:
        run_output = None

    if use_P:
        P, I_P = load_run_output(run_output)
        I_P = torch.tensor(I_P, device=device).float()
    else:
        I_P = None

    outdir = os.path.join(OUT, f"generated_text/{model_name}")
    if use_P:
        outdir = os.path.join(outdir, "I_P")
    else:
        outdir = os.path.join(outdir, "no_I_P")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count = 0
    while count < 1:#True:
        seed = random.randint(0, 999999999)
        generate_multiple_sequences(
            model, 
            prompt_ids, 
            batch_size, 
            seed, 
            export_index, 
            outdir, 
            device=device, 
            P=I_P)
        count += 1

    #tokens, token_h_pairs = generate_sequence_until_eos(model, prompt_ids, 3, 0, device)
    #I_P_tokens, I_P_token_h_pairs = generate_sequence_until_eos(model, prompt_ids, 3, 1, device, P=I_P)


#%%


#%%#################
# Computing new MI #
####################
from utils.dataset_loaders import load_processed_data
from scipy.special import softmax, kl_div
from scipy.stats import entropy
from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from utils.dataset_loaders import load_hs, load_model_eval
from evals.kl_eval import load_run_output

model_name = "gpt2-large" #"bert-base-uncased"
concept_name = "number"
#run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")
run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

hs = load_hs(concept_name, model_name)
other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)
P, I_P = load_run_output(run_output)

#%%
from evals.kl_eval import get_all_distribs, get_all_marginals, get_lemma_marginals
from evals.kl_eval import compute_kls_one_sample
h = hs[13]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, other_emb, l0_emb, l1_emb
)

distrib = base_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)

distrib = I_P_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)


#compute_kls_one_sample(h, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals )
# %%
from evals.kl_eval import compute_overall_mi
all_marginals = [
    concept_marginals["p_0_incl_other"], 
    concept_marginals["p_1_incl_other"], 
    concept_marginals["p_other_incl_other"]
]

compute_overall_mi(concept_marginals, distrib["l0"], distrib["l1"], distrib["other"])

#%%#################
# Camembert        #
####################
model = get_model("camembert-base")
tokenizer = get_tokenizer("camembert-base")
print(model)

inputs = tokenizer("La capitale de la France est <mask>", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)

# %%
from data.dataset_loaders import load_model_eval

#model_name = "gpt2-base-french"
#dataset_name = "ud_fr_gsd"
model_name = "gpt2"
dataset_name = "linzen"
model = get_model(model_name)
word_emb, l0_emb, l1_emb, lemma_prob, concept_prob = load_model_eval(dataset_name, model_name)
# %%
from scipy.stats import entropy
entropy(concept_prob)
