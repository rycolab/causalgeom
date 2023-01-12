#%%
import warnings
import logging
import os
import coloredlogs
import argparse
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC

from transformers import BertForMaskedLM

import cdopt 
from cdopt.nn import get_quad_penalty

from sklearn.model_selection import train_test_split

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

device = torch.device("cuda") if torch.cuda.is_available()\
    else torch.device("cpu")

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='ComputeTokenCounts')
    #argparser.add_argument(
    #    "--dataset", 
    #    type=str,
    #    choices=["wikipedia", "bookcorpus"],
    #    required=True,
    #    help="Dataset to extract counts from"
    #)
    argparser.add_argument(
        "--nsamples",
        type=int,
        help="Number of hidden states to compute"
    )
    argparser.add_argument(
        "--seed",
        type=int,
        help="MultiBERTs seed for tokenizer and model"
    )
    return argparser.parse_args()

#%% 
"""
args = get_args()
logging.info(args)

#TODO: FIX THIS
NSAMPLES = args.nsamples
SEED = args.seed
EXPORT_DIR = f"/cluster/scratch/cguerner/thesis_data/models/multiberts/{SEED}"

assert os.path.exists(EXPORT_DIR), "Export dir doesn't exist"

RUN_OUTPUT_DIR = os.path.join(EXPORT_DIR, TIMESTAMP)
os.mkdir(RUN_OUTPUT_DIR)

logging.info(f"Exporting hidden states from {DATASET} using "
             f"{str(NSAMPLES)} of the dataset into {RUN_OUTPUT_DIR}.")
"""
#%% TROUBLESHOOTING
#NSAMPLES = 100
SEED = 0
EXPORT_DIR = f"/cluster/scratch/cguerner/thesis_data/models/multiberts/{SEED}"

assert os.path.exists(EXPORT_DIR), "Export dir doesn't exist"

RUN_OUTPUT_DIR = os.path.join(EXPORT_DIR, TIMESTAMP)
os.mkdir(RUN_OUTPUT_DIR)

#TRAIN_SIZE = 100
#VAL_SIZE = 10
#TEST_SIZE = 10

#%%
LM = BertForMaskedLM.from_pretrained(
    f'google/multiberts-seed_{SEED}', 
    cache_dir="/cluster/scratch/cguerner/thesis_data/hf_cache", 
    is_decoder=False
)

#%% Ws
word_embeddings = LM.bert.embeddings.word_embeddings.weight
bias = LM.cls.predictions.decoder.bias
x_w = torch.cat(
    (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()

#%% Hs
#h_wiki = np.load("/cluster/scratch/cguerner/thesis_data/h_matrices/multiberts_0_wikipedia_20221128_184647.npy")
h_book = np.load("/cluster/scratch/cguerner/thesis_data/h_matrices/multiberts_0_bookcorpus_20221201_231959.npy")

#h = np.concatenate([h_wiki, h_book], axis=0)
h = h_book

#%%
def add_constant_dim(hidden_states):
    constant_dim = np.ones((hidden_states.shape[0], 1))
    hidden_states_wones = np.hstack((hidden_states, constant_dim))
    return hidden_states_wones

h = add_constant_dim(h)

#%%
#TODO: make bigger sample version
token_ids, log_unigram_probs = np.load(
    "/cluster/home/cguerner/freqexp/data/unigram_distribs/multibert.npy"
)

#%%
#TODO h train val test split
h_train, h_valtest = train_test_split(h, train_size=.01, test_size=.005)
h_val, h_test = train_test_split(h_valtest, train_size=.5)

#%%
class BiaffineDataset(Dataset, ABC):
    def __init__(self, x_h, x_w, log_unigram_probs):
        self.x_h = x_h
        self.x_w = x_w
        self.log_unigram_probs = log_unigram_probs

        self.h_shape = self.x_h.shape[0]
        self.w_shape = self.x_w.shape[0]
        self.n_instances = self.h_shape * self.w_shape

    def __len__(self):
        return self.n_instances

    def __get_indices(self, index):
        index_h = int(index / self.w_shape)
        index_w = index % self.w_shape
        return index_h, index_w

    def __getitem__(self, index):
        if type(index) == int:
            index_h, index_w = self.__get_indices(index)
            return (
                self.x_h[index_h], 
                self.x_w[index_w], 
                self.log_unigram_probs[index_w]
            )
        elif type(index) == list:
            indices_h = []
            indices_w = []
            for i in index:
                index_h, index_w = self.__get_indices(index)
                indices_h.append(index_h)
                indices_w.append(index_w)
            return (
                self.x_h[indices_h], 
                self.x_w[indices_w], 
                self.log_unigram_probs[indices_w]
            )
        else:
            return TypeError("Index has to be int or list of ints.")

#%%
train_set = BiaffineDataset(h_train, x_w, log_unigram_probs)
val_set = BiaffineDataset(h_val, x_w, log_unigram_probs)
test_set = BiaffineDataset(h_test, x_w, log_unigram_probs)

#%%
from torch.utils.data import DataLoader, SubsetRandomSampler, BatchSampler
BATCH_SIZE = 1000

train_set = BiaffineDataset(h_train, x_w, log_unigram_probs)

train_sampler = BatchSampler(
    SubsetRandomSampler(range(len(train_set))), 
    batch_size=BATCH_SIZE,
    drop_last=True
)

#%%
train_sampler = BatchSampler(
    SubsetRandomSampler(range(12)),
    batch_size=3,
    drop_last=True
)

#%%

train_loader = DataLoader(
    train_set,
    batch_size=1000,
    shuffle=True
)
val_loader = DataLoader(
    val_set,
    batch_size=1000
)
test_loader = DataLoader(
    test_set,
    batch_size=1000
)

#%%
from tqdm import tqdm

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    nbatches = len(train_loader)

    for i, data in enumerate(pbar := tqdm(train_loader)):
        pbar.set_description(f"Epoch {epoch_index}, batches")
        # Every data instance is an input + label pair
        x_h, x_w, labels  = data
        x_w = x_w.double()
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model.forward(x_h, x_w)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels) + get_quad_penalty(model)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if i > NBATCHES_TRAIN:
            break

        # Gather data and report

        running_loss += loss.item()
        #if i % int(nbatches*.1) ==  int(nbatches*.1) - 1:
        #    last_loss = running_loss / int(nbatches*.1) # loss per batch
        #    print('  batch {} loss: {}'.format(i + 1, last_loss))
        #    #tb_x = epoch_index * len(training_loader) + i + 1
        #    #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
         #   running_loss = 0.

    return running_loss/(i+1)

#%%
import models.models as models
from models.models import Biaffine
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
dim = x_w.shape[1]
model = Biaffine(dim)
loss_fn = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
output_folder = "trained_models/biaffine/"
epoch_number = 0


NBATCHES_TRAIN = 1
NBATCHES_VAL = 1

EPOCHS = 1

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    logging.info('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(test_loader):
        v_x_h, v_x_w, vlabels = vdata
        v_x_w = v_x_w.double()
        voutputs = model(v_x_h, v_x_w)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()

        if i > NBATCHES_VAL:
            break

    avg_vloss = running_vloss / (i + 1)
    logging.info('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    scheduler.step(avg_vloss)
    # Log the running loss averaged per batch
    # for both training and validation
    #writer.add_scalars('Training vs. Validation Loss',
    #                { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                epoch + 1)
    #writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss and epoch > int(.9*EPOCHS):
        best_vloss = avg_vloss
        model_path = os.path.join(
            output_folder, 'model_{}_{}'.format(timestamp, epoch)
        )
        torch.save(model.state_dict(), model_path)

#%%
saved_model = Biaffine(dim)
model_path = "trained_models/biaffine/model_20221117_163817_29"
saved_model.load_state_dict(torch.load(model_path))

#%%
weights = saved_model.linear.weight.detach().numpy().flatten()

#%%
import seaborn as sns

sns.barplot(x = np.arange(weights.shape[0], dtype=int), y = weights.flatten())
