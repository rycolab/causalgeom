import os

#TODO: remove TIANYU_SHARED here and throughout repo
MAIN = "/cluster/work/cotterell/cguerner/usagebasedprobing/"
TIANYU_SHARED = "/cluster/work/cotterell/tianyu_shared/"
OUT = os.path.join(MAIN, "out")
DATASETS = os.path.join(MAIN, "data")
RESULTS = os.path.join(MAIN, "results")
TIANYU_RESULTS = os.path.join(TIANYU_SHARED, "results")
MODELS = os.path.join(MAIN, "models")
HF_CACHE = "../hf_cache"
AUTH_TOKEN_PATH = os.path.join(MAIN, "auth_token.txt")


#BERT_SYNTAX_LINZEN_GOLDBERG = "../bert-syntax/lgd_dataset.tsv"

LINZEN_RAW = os.path.join(DATASETS, "raw/linzen/agr_50_mostcommon_10k.tsv")
LINZEN_VOCAB = os.path.join(DATASETS, "raw/linzen/wiki.vocab")

#UNIMORPH_ENG = "/cluster/work/cotterell/cguerner/eng/eng"

UD_FRENCH_GSD = "/cluster/work/cotterell/cguerner/UD_French-GSD"
UD_FRENCH_ParTUT = "/cluster/work/cotterell/cguerner/UD_French-ParTUT"
UD_FRENCH_Rhapsodie = "/cluster/work/cotterell/cguerner/UD_French-Rhapsodie"

FR_DATASETS = ["ud_fr_gsd", "ud_fr_partut", "ud_fr_rhapsodie"]