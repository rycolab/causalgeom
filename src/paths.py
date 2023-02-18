import os

MAIN = "/cluster/work/cotterell/cguerner/usagebasedprobing/"

OUT = os.path.join(MAIN, "out")
DATASETS = os.path.join(MAIN, "datasets")
HF_CACHE = "../hf_cache"


BERT_SYNTAX_LINZEN_GOLDBERG = "../bert-syntax/lgd_dataset.tsv"

LINZEN_RAW = "/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/raw/linzen/agr_50_mostcommon_10k.tsv"
LINZEN_VOCAB = '/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/raw/linzen/wiki.vocab'
LINZEN_PREPROCESSED = "/cluster/work/cotterell/cguerner/usagebasedprobing/datasets/preprocessed/linzen_preprocessed.tsv"

UNIMORPH_ENG = "/cluster/work/cotterell/cguerner/eng/eng"