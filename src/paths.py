import os

MAIN = "/cluster/home/tialiu/tianyu_c/usagebasedprobing/"

OUT = os.path.join(MAIN, "out")
DATASETS = os.path.join(MAIN, "data")
RESULTS = os.path.join(MAIN, "results")
MODELS = os.path.join(MAIN, "models")
HF_CACHE = "../hf_cache"
AUTH_TOKEN_PATH = os.path.join(MAIN, "auth_token.txt")


BERT_SYNTAX_LINZEN_GOLDBERG = "../bert-syntax/lgd_dataset.tsv"

LINZEN_RAW = os.path.join(DATASETS, "raw/linzen/agr_50_mostcommon_10k.tsv")
LINZEN_VOCAB = os.path.join(DATASETS, "raw/linzen/wiki.vocab")

UNIMORPH_ENG = "/cluster/home/tialiu/tianyu_c/eng/eng"

UD_FRENCH_GSD = "/cluster/home/tialiu/tianyu_c/UD_French-GSD"
UD_FRENCH_ParTUT = "/cluster/home/tialiu/tianyu_c/UD_French-ParTUT"
UD_FRENCH_Rhapsodie = "/cluster/home/tialiu/tianyu_c/UD_French-Rhapsodie"

FR_DATASETS = ["ud_fr_gsd", "ud_fr_partut", "ud_fr_rhapsodie"]