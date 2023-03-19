import os
import sys

#sys.path.append('..')
sys.path.append('./src/')

LINZEN_PREPROCESSED = os.path.join(DATASETS, "preprocessed/linzen_preprocessed.tsv")
UD_FRENCH_GSD = os.path.join(DATASETS, "preprocessed/ud/fr/gsd")


#%% LINZEN LOADERS
def load_linzen_ar():
    data = []
    with open(LINZEN_PREPROCESSED) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            unmasked_text = line[1]
            verb = line[3]
            iverb = line[4]
            verb_pos = line[5]
            vindex = int(line[6])
            if vindex > 0:
                verb = " " + verb
                iverb = " " + iverb
            pre_verb_text = " ".join(unmasked_text.split(" ")[:vindex])
            verb_text = " ".join(unmasked_text.split(" ")[:(vindex+1)])
            iverb_text = " ".join(unmasked_text.split(" ")[:vindex] + [iverb])
            sample = dict(
                pre_tgt_text=pre_verb_text,
                fact_text=verb_text,
                foil_text=iverb_text,
                fact=verb,
                foil=iverb,
                tgt_label=verb_pos
            )
            data.append(sample)
    return data

def load_linzen_masked():
    data = []
    with open(LINZEN_PREPROCESSED) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            #masked = line[2].replace("***mask***", "[MASK]")
            masked = line[2]
            #mask_index = masked.split().index("[MASK]")
            sample = dict(
                masked=masked,
                #mask_index=mask_index,
                fact=line[3],
                foil=line[4],
                tgt_label=line[5]
            )
            data.append(sample)
    return data

def load_linzen(model_type):
    if model_type == "ar":
        return load_linzen_ar()
    elif model_tupe == "masked":
        return load_linzen_masked()
    else:
        raise ValueError("Invalid model type for Linzen")


#%% UD 


#%%
def get_model_type(model_name):
    if model_name == "gpt2":
        return "ar"
    elif model_name == "bert-base-uncased":
        return "masked"
    else: 
        raise ValueError(f"Model {model_name} not supported")

def load_dataset(dataset_name, model_name, split=None):
    """ Dataset name: ["linzen", "ud_fr_gsd"]
    Model name: ["gpt2", "bert-base-uncased"]
    Split: ["train", "dev", "test"] (only for UD)
    """
    model_type = get_model_type(model_name)
    if dataset_name == "linzen":
        return load_linzen(model_type)
    #TODO: complete this second if 
    elif dataset_name == "ud_fr_gsd":
        assert split in ["train", "dev", "test"], "Must specify split"
        return os.path.join(UD_FRENCH_GSD, f"{split}.csv")
    else:
        raise ValueError("invalid dataset name")