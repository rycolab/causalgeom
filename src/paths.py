import os

MAIN = "/cluster/work/cotterell/cguerner/usagebasedprobing/"

OUT = os.path.join(MAIN, "out")
DATASETS = os.path.join(MAIN, "data")
HF_CACHE = "../hf_cache"


BERT_SYNTAX_LINZEN_GOLDBERG = "../bert-syntax/lgd_dataset.tsv"

LINZEN_RAW = os.path.join(DATASETS, "raw/linzen/agr_50_mostcommon_10k.tsv")
LINZEN_VOCAB = os.path.join(DATASETS, "raw/linzen/wiki.vocab")

UNIMORPH_ENG = "/cluster/work/cotterell/cguerner/eng/eng"

UD_FRENCH_GSD = "/cluster/work/cotterell/cguerner/UD_French-GSD"
UD_FRENCH_ParTUT = "/cluster/work/cotterell/cguerner/UD_French-ParTUT"
UD_FRENCH_Rhapsodie = "/cluster/work/cotterell/cguerner/UD_French-Rhapsodie"

FR_DATASETS = ["ud_fr_gsd", "ud_fr_partut", "ud_fr_rhapsodie"]