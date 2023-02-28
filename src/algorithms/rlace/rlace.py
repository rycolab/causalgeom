import sys
import os
import time
import random
#import ipdb
import warnings
import logging
import coloredlogs
import wandb

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import SGDClassifier
from torch.optim import SGD, Adam
import sklearn

from abc import ABC

#sys.path.append('..')
sys.path.append('./src/')

from classifiers.classifiers import BinaryParamFreeClf, BinaryParamFreeClfTwoPs
#import functionals

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

EVAL_CLF_PARAMS = {
    "loss": "log_loss", 
    "tol": 1e-4, 
    "iters_no_change": 15, 
    "alpha": 1e-4, 
    "max_iter": 25000
}
NUM_CLFS_IN_EVAL = 3 # change to 1 for large dataset / high dimensionality

def init_classifier():

    return SGDClassifier(
        loss=EVAL_CLF_PARAMS["loss"], 
        fit_intercept=True, 
        max_iter=EVAL_CLF_PARAMS["max_iter"], 
        tol=EVAL_CLF_PARAMS["tol"], 
        n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
        n_jobs=32, 
        alpha=EVAL_CLF_PARAMS["alpha"]
    )
                        
def symmetric(X):
    X.data = 0.5 * (X.data + X.data.T)
    return X

def run_validation(X_train, y_train, X_dev, y_dev, P, rank):
    __, I_P_svd = get_projection(P, rank)
    
    loss_vals = []
    accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train@I_P_svd, y_train)
        y_pred = clf.predict_proba(X_dev@I_P_svd)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(X_dev@I_P_svd, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]
"""
def get_score_param_free(X_dev, U_dev, y_dev, P, rank, device):
    P_svd = torch.Tensor(get_projection(P, rank)).to(device)
    clf = BinaryParamFreeClf(X_dev, U_dev, y_dev, P_svd, device)

    return clf.loss_P(), clf.score_P()

def get_score_param_free_twoPs(X_dev, U_dev, y_dev, Pu, Ph, rank, device):
    Pu_svd = torch.Tensor(get_projection(Pu, rank)).to(device)
    Ph_svd = torch.Tensor(get_projection(Ph, rank)).to(device)
    
    clf = BinaryParamFreeClfTwoPs(X_dev, U_dev, y_dev, Pu_svd, Ph_svd, device)

    return clf.loss_P(), clf.score_P()
"""
def solve_constraint(lambdas, d=1):
    def f(theta):
        return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
        return return_val

    theta_min, theta_max = max(lambdas), min(lambdas) - 1
    assert f(theta_min) * f(theta_max) < 0

    mid = (theta_min + theta_max) / 2
    tol = 1e-4
    iters = 0

    while iters < 25:

        mid = (theta_min + theta_max) / 2

        if f(mid) * f(theta_min) > 0:

            theta_min = mid
        else:
            theta_max = mid
        iters += 1

    lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
    # if (theta_min-theta_max)**2 > tol:
    #    print("didn't converge", (theta_min-theta_max)**2)
    return lambdas_plus

def get_majority_acc(y):

    from collections import Counter
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    maj = max(fracts)
    return maj

def get_majority_acc_entropy(y):

    from collections import Counter
    import scipy
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    maj = max(fracts)
    ent = scipy.stats.entropy(fracts)
    return maj, ent

def get_entropy(y):

    from collections import Counter
    import scipy
    
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    return scipy.stats.entropy(fracts)
    

def get_projection(P, rank):
    D,U = np.linalg.eigh(P)
    U = U.T
    W = U[-rank:]
    P_final = W.T @ W
    I_P_final = np.eye(P.shape[0]) - P_final
    return P_final, I_P_final

def prepare_output(P_loss, P_acc, rank, best_acc, best_loss):
    P_loss_svd, I_P_loss_svd = get_projection(P_loss, rank)
    P_acc_svd, I_P_acc_svd = get_projection(P_acc, rank)
    return {
        "best_loss": best_loss, 
        "best_acc": best_acc, 
        "P": P_loss_svd,
        "I_P": I_P_loss_svd,
        "P_acc": P_acc_svd,
        "I_P_acc": I_P_acc_svd
    }

"""
def prepare_output_twoPs(Pu, Ph, rank, best_score, best_loss):
    Pu_final = get_projection(Pu, rank)
    Ph_final = get_projection(Ph, rank)
    return {"best_score": best_score, 
            "best_loss": best_loss, 
            "Pu_before_svd": np.eye(Pu.shape[0]) - Pu, 
            "Pu": Pu_final, 
            "P_before_svd": np.eye(Ph.shape[0]) - Ph, 
            "P": Ph_final}
"""

def get_default_predictor(X_train, y_train, device):
    #TODO: change this to just X shape and num_labels
    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        return torch.nn.Linear(X_train.shape[1], 1).to(device)
    else:
        return torch.nn.Linear(X_train.shape[1], num_labels).to(device)

def solve_adv_game(X_train, y_train, X_dev, y_dev, 
                   predictor=None, rank=1, device="cpu", out_iters=75000, 
                   in_iters_adv=1, in_iters_clf=1, epsilon=0.0015, 
                   batch_size=128, evaluate_every=1000, 
                   optimizer_class=SGD, 
                   optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}, 
                   optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-4}, 
                   scheduler_class=None, scheduler_params_P=None,
                   torch_outfile=None, wb=False, wb_run=None):
    """
    :param X: The input (np array)
    :param Y: the lables (np array)
    :param X_dev: Dev set (np array)
    :param Y_dev: Dev labels (np array)
    :param rank: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion .Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evaluate_every: After how many batches to evaluate the current adversary.
    :param optimizer_class: SGD/Adam etc.
    :param optimizer_params: the optimizer's params (as a dict)
    :return:
    """

    def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
        I = torch.eye(X_train.shape[1]).to(device)
        bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y)
        if optimize_P:
            bce = -bce
        return bce

    if predictor == None:
        predictor = get_default_predictor(X_train, y_train, device)

    X_torch = torch.tensor(X_train).float()
    y_torch = torch.tensor(y_train).float()
    #X_torch = torch.tensor(X_train).float()
    #y_torch = torch.tensor(y_train).float()

    #SUBSET OF TRAIN SET FOR TRAINING CLF DURING VALIDATION
    VAL_TRAIN_SIZE = 15000
    val_train_idx = np.arange(0, X_torch.shape[0])
    np.random.shuffle(val_train_idx)
    X_val_train = X_torch[val_train_idx[:VAL_TRAIN_SIZE]]
    y_val_train = y_torch[val_train_idx[:VAL_TRAIN_SIZE]]

    X_torch = X_torch.to(device)
    y_torch = y_torch.to(device)

    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        #predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        y_torch = y_torch.float()
    else:
        #predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        y_torch = y_torch.long()

    P = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    P.requires_grad = True

    optimizer_predictor = optimizer_class(predictor.parameters(), **optimizer_params_predictor)
    optimizer_P = optimizer_class([P],**optimizer_params_P)

    assert scheduler_class is not None

    scheduler_P = scheduler_class(optimizer_P, **scheduler_params_P)

    maj, label_entropy = get_majority_acc_entropy(y_train)
    best_P, best_P_acc, best_acc, best_loss = None, None, 1, -1

    if wb:
        if wb_run is None:
            wb_run = 0

    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    for i in pbar:

        for j in range(in_iters_adv):
            P = symmetric(P)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]
            #X_batch, y_batch = X_torch[idx[:batch_size]].to(device), y_torch[idx[:batch_size]].to(device)

            loss_P = get_loss_fn(X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=True)
            loss_P.backward()
            optimizer_P.step()

            # project

            with torch.no_grad():
                D, U = torch.linalg.eigh(symmetric(P).detach().cpu())
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=rank)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                U = U.to(device)
                P.data = U @ D @ U.T

        for j in range(in_iters_clf):
            optimizer_predictor.zero_grad()
            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]
            #X_batch, y_batch = X_torch[idx[:batch_size]].to(device), y_torch[idx[:batch_size]].to(device)

            loss_predictor = get_loss_fn(X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=False)
            loss_predictor.backward()
            optimizer_predictor.step()

        if i % evaluate_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            loss_val, acc_val = run_validation(X_val_train, y_val_train, X_dev, y_dev, P.detach().cpu().numpy(), rank)
            #TODO: probably want to pick best_score and best_loss in the same if statement (evaluate on one)
            if loss_val > best_loss:#if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            if np.abs(acc_val - maj) < np.abs(best_acc - maj):
                best_P_acc, best_acc = symmetric(P).detach().cpu().numpy().copy(), acc_val

            scheduler_P.step(loss_val)

            if wb:
                wandb.log({f"diag_rlace/val/{wb_run}/loss": loss_val, 
                            f"diag_rlace/val/{wb_run}/acc": acc_val,
                            f"diag_rlace/val/{wb_run}/best_loss": best_loss,
                            f"diag_rlace/val/{wb_run}/best_acc": best_acc,
                            f"diag_rlace/val/{wb_run}/lr": optimizer_P.param_groups[0]['lr']})
            
            # update progress bar
            pbar.set_description(
                "{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%;"
                " Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(
                    i, out_iters, acc_val * 100, best_acc * 100, maj * 100, 
                    np.abs(best_acc - maj) * 100, best_loss, loss_val
                )
            )
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        #if i > 1 and np.abs(best_score - maj) < epsilon:
        #if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
        #    break
    output = prepare_output(
        best_P, best_P_acc, rank, best_acc, best_loss
    )
    if torch_outfile is not None:
        torch.save(
            {"model_state_dict": predictor.state_dict(), 
            "optimizer_state_dict": optimizer_predictor.state_dict()}, 
            torch_outfile
        )
    return output


if __name__ == "__main__":
    
    #random.seed(0)
    #np.random.seed(0)

    # create a synthetic dataset
    n, dim = 15000, 200
    num_classes = 2
    
    X = np.random.randn(n, dim)
    y = np.random.randint(low = 0, high = num_classes, size = n) #(np.random.rand(n) > 0.5).astype(int)

    X[:, 0] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.3 * y
    X[:, 1] = (y + np.random.randn(*y.shape) * 0.1) ** 2 - 0.7 * y
    X[:, 2] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.5 * y + np.random.randn(*y.shape) * 0.2
    X[:, 3] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.7 * y + np.random.randn(*y.shape) * 0.1
    X[:, 4] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.8 * y + np.random.randn(*y.shape) * 0.1
    X[:, 5] = (y + np.random.randn(*y.shape) * 0.25) ** 2 - 0.2 * y + np.random.randn(*y.shape) * 0.1
    mixing_matrix = 1e-2*np.random.randn(dim, dim)
    X = X @ mixing_matrix
    
    l_train = int(0.6*n)
    X_train, y_train = X[:l_train], y[:l_train]
    X_dev, y_dev = X[l_train:], y[l_train:]

    # arguments
    num_iters = 50000
    rank=1
    optimizer_class = torch.optim.SGD
    optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
    optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
    epsilon = 0.001 # stop 0.1% from majority acc
    batch_size = 256

    output = solve_adv_game(X_train, y_train, X_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size)
    
    # train a classifier
    
    P_svd = output["P"]
    P_before_svd = output["P_before_svd"]
    svm = init_classifier()
                        
    svm.fit(X_train[:] , y_train[:])
    score_original = svm.score(X_dev, y_dev)
    
    svm = init_classifier()
    svm.fit(X_train[:] @ P_before_svd , y_train[:])
    score_projected_no_svd = svm.score(X_dev @ P_before_svd, y_dev)
    
    svm = init_classifier()
    svm.fit(X_train[:] @ P_svd , y_train[:])
    score_projected_svd_dev = svm.score(X_dev @ P_svd, y_dev)
    score_projected_svd_train = svm.score(X_train @ P_svd, y_train)
    maj_acc_dev = get_majority_acc(y_dev)
    maj_acc_train = get_majority_acc(y_train)
    
    print("===================================================")
    print("Original Acc, dev: {:.3f}%; Acc, projected, no svd, dev: {:.3f}%; Acc, projected+SVD, train: {:.3f}%; Acc, projected+SVD, dev: {:.3f}%".format(
        score_original*100, score_projected_no_svd*100, score_projected_svd_train*100, score_projected_svd_dev*100))    
    print("Majority Acc, dev: {:.3f} %".format(maj_acc_dev*100))
    print("Majority Acc, train: {:.3f} %".format(maj_acc_train*100))
    print("Gap, dev: {:.3f} %".format(np.abs(maj_acc_dev - score_projected_svd_dev)*100))
    print("Gap, train: {:.3f} %".format(np.abs(maj_acc_train - score_projected_svd_train)*100))
    print("===================================================")
    eigs_before_svd, _ = np.linalg.eigh(P_before_svd)
    print("Eigenvalues, before SVD: {}".format(eigs_before_svd[:]))
    
    eigs_after_svd, _ = np.linalg.eigh(P_svd)
    print("Eigenvalues, after SVD: {}".format(eigs_after_svd[:]))
    
    eps = 1e-6
    assert np.abs( (eigs_after_svd > eps).sum() -  (dim - rank) ) < eps
    
