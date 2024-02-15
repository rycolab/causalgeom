#%%
from utils.lm_loaders import get_tokenizer, get_model
from utils.auth_utils import load_auth_token

#%%
#tokenizer = get_tokenizer("llama2", token=load_auth_token())
model = get_model("llama2", token=load_auth_token())

#%%
