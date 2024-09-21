import os

# ONBOARDING STEPS:
# 1. Set MAIN path to the folder where you cloned the repository
MAIN = ""
OUT = os.path.join(MAIN, "out")
DATASETS = os.path.join(MAIN, "data")
RESULTS = os.path.join(MAIN, "results")
MODELS = os.path.join(MAIN, "models")
HF_CACHE = "../hf_cache"

# 2. Set the path to your huggingface authentication token
AUTH_TOKEN_PATH = os.path.join(MAIN, "auth_token.txt")

# 3. Set paths to where you downloaded the raw Linzen data files
LINZEN_RAW = os.path.join(DATASETS, "raw/linzen/agr_50_mostcommon_10k.tsv")
LINZEN_VOCAB = os.path.join(DATASETS, "raw/linzen/wiki.vocab")

# 4. Set paths to where you cloned the TreeBank repositories
UD_REPOS_MAIN = ""
UD_FRENCH_GSD = os.path.join(
    UD_REPOS_MAIN, "UD_French-GSD"
)
UD_FRENCH_ParTUT = os.path.join(
    UD_REPOS_MAIN, "UD_French-ParTUT"
)
UD_FRENCH_Rhapsodie = os.path.join(
    UD_REPOS_MAIN, "UD_French-Rhapsodie"
)

# Don't modify this list below, unless you want to add a FR Treebank
FR_DATASETS = ["ud_fr_gsd", "ud_fr_partut", "ud_fr_rhapsodie"]