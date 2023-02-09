import numpy as np
import tqdm
import torch
from sklearn.linear_model import SGDClassifier
import time
from torch.optim import SGD, Adam
import random
import sklearn
import ipdb
import warnings
import logging
import coloredlogs
import wandb

from torch.utils.data import DataLoader, Dataset
from abc import ABC

from classifiers.classifiers import BinaryParamFreeClf, BinaryParamFreeClfTwoPs
import functionals

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
    P_svd = get_projection(P, rank)
    
    loss_vals = []
    accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train@P_svd, y_train)
        y_pred = clf.predict_proba(X_dev@P_svd)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(X_dev@P_svd, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]

def get_score_param_free(X_dev, U_dev, y_dev, P, rank, device):
    P_svd = torch.Tensor(get_projection(P, rank)).to(device)
    clf = BinaryParamFreeClf(X_dev, U_dev, y_dev, P_svd, device)

    return clf.loss_P(), clf.score_P()

def get_score_param_free_twoPs(X_dev, U_dev, y_dev, Pu, Ph, rank, device):
    Pu_svd = torch.Tensor(get_projection(Pu, rank)).to(device)
    Ph_svd = torch.Tensor(get_projection(Ph, rank)).to(device)
    
    clf = BinaryParamFreeClfTwoPs(X_dev, U_dev, y_dev, Pu_svd, Ph_svd, device)

    return clf.loss_P(), clf.score_P()

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
    P_final = np.eye(P.shape[0]) - W.T @ W
    return P_final

def prepare_output(P_loss, P_acc, rank, best_acc, best_loss):
    P_loss_svd = get_projection(P_loss, rank)
    P_acc_svd = get_projection(P_acc, rank)
    return {
        "best_loss": best_loss, 
        "best_acc": best_acc, 
        "P_before_svd": np.eye(P_loss.shape[0]) - P_loss, 
        "P": P_loss_svd, 
        "P_acc_before_svd": np.eye(P_acc.shape[0]) - P_acc, 
        "P_acc": P_acc_svd
    }

def prepare_output_twoPs(Pu, Ph, rank, best_score, best_loss):
    Pu_final = get_projection(Pu, rank)
    Ph_final = get_projection(Ph, rank)
    return {"best_score": best_score, 
            "best_loss": best_loss, 
            "Pu_before_svd": np.eye(Pu.shape[0]) - Pu, 
            "Pu": Pu_final, 
            "P_before_svd": np.eye(Ph.shape[0]) - Ph, 
            "P": Ph_final}


def get_default_predictor(X_train, y_train, device):
    #TODO: change this to just X shape and num_labels
    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        return torch.nn.Linear(X_train.shape[1], 1).to(device)
    else:
        return torch.nn.Linear(X_train.shape[1], num_labels).to(device)

def solve_adv_game(X_train, y_train, X_dev, y_dev, predictor=None, rank=1, device="cpu", out_iters=75000, 
    in_iters_adv=1, in_iters_clf=1, epsilon=0.0015, batch_size=128, evaluate_every=1000, 
    optimizer_class=SGD,  optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}, 
    optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-4}, torch_outfile=None, wb=False):
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

    X_torch = torch.tensor(X_train).float().to(device)
    y_torch = torch.tensor(y_train).float().to(device)

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

    maj, label_entropy = get_majority_acc_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    count_examples = 0
    best_P, best_P_acc, best_acc, best_loss = None, None, 1, -1

    for i in pbar:

        for j in range(in_iters_adv):
            P = symmetric(P)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

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

            loss_predictor = get_loss_fn(X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=False)
            loss_predictor.backward()
            optimizer_predictor.step()
            count_examples += batch_size

        if i % evaluate_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            loss_val, acc_val = run_validation(X_train, y_train, X_dev, y_dev, P.detach().cpu().numpy(), rank)
            if wb:
                wandb.log({"diag_rlace/val/loss": loss_val, 
                            "diag_rlace/val/acc": acc_val})
            #TODO: probably want to pick best_score and best_loss in the same if statement (evaluate on one)
            if loss_val > best_loss:#if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            if np.abs(acc_val - maj) < np.abs(best_acc - maj):
                best_P_acc, best_acc = symmetric(P).detach().cpu().numpy().copy(), acc_val
                
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

def solve_adv_game_param_free(X_train, U_train, y_train, X_dev, U_dev, y_dev, version,
    rank=1, device="cpu", out_iters=75000, in_iters_adv=1, in_iters_clf=1, 
    epsilon=0.0015, batch_size=128, evaluate_every=1000, 
    optimizer_class=SGD,  optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}):
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

    def get_loss_fn(X, U, y, P, bce_loss_fn, optimize_P=False):
        I = torch.eye(X_train.shape[1]).to(device)
        #TODO: did NOT include torch.nn.Sigmoid here because binary clf
        # NOTE: seems like the param version outputs predictor() and those values look
        # like after applying nn.Sigmoid
        #pred = torch.sum(torch.mm(X @ (I-P),torch.t(U)), dim = 1)
        pred = ((X @ (I-P)) * (U)).sum(-1)
        bce = bce_loss_fn(pred, y)
        if optimize_P:
            bce = -bce
        return bce

    if version == 'positively_functional':
        functional = functionals.PositivelyFunctional(
            get_loss_fn, get_score_param_free
        )
    elif version == 'negatively_functional':
        functional = functionals.NegativelyFunctional(
            get_loss_fn, get_score_param_free
        )
    elif version == 'original':
        functional = functionals.OriginalFunctional(
            get_loss_fn, get_score_param_free
        )
    else:
        ValueError(f'Received an invalid version ({version}) of functional RLACE')

    logging.info("Loading data to GPU")
    X_train = torch.from_numpy(X_train).float()
    U_train = torch.from_numpy(U_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_dev = torch.from_numpy(X_dev).float().to(device)
    U_dev = torch.from_numpy(U_dev).float().to(device)
    #y_dev = torch.tensor(y_dev).float()

    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        #predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        #y_train = y_train
        #y_dev = y_dev
    else:
        #predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        #y_train = y_train
        #y_dev = y_dev

    P = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    P.requires_grad = True

    #optimizer_predictor = optimizer_class(predictor.parameters(), **optimizer_params_predictor)
    optimizer_P = optimizer_class([P],**optimizer_params_P)
    
    maj, label_entropy = get_majority_acc_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    val_results = []
    for i in pbar:

        for j in range(in_iters_adv):
            P = symmetric(P)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_train.shape[0])
            np.random.shuffle(idx)
            X_batch, U_batch, y_batch = X_train[idx[:batch_size]].to(device), U_train[idx[:batch_size]].to(device), y_train[idx[:batch_size]].to(device)

            loss_P = functional.get_loss(
                X_batch, U_batch, y_batch, 
                symmetric(P), bce_loss_fn,
                device
            )

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
            #optimizer_predictor.zero_grad()
            #idx = np.arange(0, X_train_torch.shape[0])
            #np.random.shuffle(idx)
            
            #X_batch, U_batch, y_batch = X_train_torch[idx[:batch_size]], U_train_torch[idx[:batch_size]], y_train_torch[idx[:batch_size]]

            #loss_predictor = get_loss_fn(X_batch, U_batch, y_batch, symmetric(P), bce_loss_fn, optimize_P=False)
            #loss_predictor.backward()
            #optimizer_predictor.step()
            count_examples += batch_size

        if i % evaluate_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            #loss_val, score = get_score_param_free(X_dev, U_dev, y_dev, P.detach().cpu(), rank, device)
            loss_val, score = functional.get_loss_and_score(
                X_dev, U_dev, y_dev, P, rank, device
            )
            val_results.append((loss_val, score))
            if functional.is_best_loss(loss_val):#if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            #if np.abs(score - maj) < np.abs(best_score - maj):
            if functional.is_best_acc(score):
                best_P_score, best_score = symmetric(P).detach().cpu().numpy().copy(), score
                
            # update progress bar
            
            pbar.set_description(
                "{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(
                    i, out_iters, score * 100, best_score * 100, 
                    maj * 100, np.abs(best_score - maj) * 100, best_loss, loss_val))
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        #if i > 1 and np.abs(best_score - maj) < epsilon:
        #if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
        #            break
    output = prepare_output(best_P,best_P_score,rank,best_score,best_loss,val_results)
    return output

def solve_adv_game_param_free_twoPs(X_train, U_train, y_train, X_dev, U_dev, y_dev, 
    rank=1, device="cpu", out_iters=75000, in_iters_adv=1, in_iters_clf=1, 
    epsilon=0.0015, batch_size=128, evaluate_every=1000, 
    optimizer_class=SGD,  optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}):
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

    def get_loss_fn(X, U, y, Pu, Ph, bce_loss_fn, optimize_Ph=False):
        I = torch.eye(X_train.shape[1]).to(device)
        pred = ((X @ (I-Ph)) * (U @ (I-Pu))).sum(-1)
        bce = bce_loss_fn(pred, y)
        if optimize_Ph:
            bce = -bce
        return bce

    logging.info("Loading data to GPU")
    X_train = torch.from_numpy(X_train).float()
    U_train = torch.from_numpy(U_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_dev = torch.from_numpy(X_dev).float().to(device)
    U_dev = torch.from_numpy(U_dev).float().to(device)
    #y_dev = torch.tensor(y_dev).float()

    num_labels = len(set(y_train.tolist()))
    if num_labels == 2:
        #predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        #y_train = y_train
        #y_dev = y_dev
    else:
        #predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        #y_train = y_train
        #y_dev = y_dev

    Pu = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    Ph = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    Pu.requires_grad = True
    Ph.requires_grad = True

    #optimizer_predictor = optimizer_class(predictor.parameters(), **optimizer_params_predictor)
    optimizer_Pu = optimizer_class([Pu],**optimizer_params_P)
    optimizer_Ph = optimizer_class([Ph],**optimizer_params_P)
    
    maj, label_entropy = get_majority_acc_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    for i in pbar:

        for j in range(in_iters_adv):
            Pu = symmetric(Pu)
            Ph = symmetric(Ph)
            optimizer_Pu.zero_grad()
            optimizer_Ph.zero_grad()


            idx = np.arange(0, X_train.shape[0])
            np.random.shuffle(idx)
            X_batch, U_batch, y_batch = X_train[idx[:batch_size]].to(device), U_train[idx[:batch_size]].to(device), y_train[idx[:batch_size]].to(device)

            loss_P = get_loss_fn(X_batch, U_batch, y_batch, symmetric(Pu), symmetric(Ph), bce_loss_fn, optimize_Ph=True)
            loss_P.backward()
            optimizer_Ph.step()

            # project

            with torch.no_grad():
                D, U = torch.linalg.eigh(symmetric(Ph).detach().cpu())
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=rank)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                U = U.to(device)
                Ph.data = U @ D @ U.T

        for j in range(in_iters_clf):
            Pu = symmetric(Pu)
            Ph = symmetric(Ph)
            optimizer_Pu.zero_grad()
            optimizer_Ph.zero_grad()

            loss_P = get_loss_fn(X_batch, U_batch, y_batch, symmetric(Pu), symmetric(Ph), bce_loss_fn, optimize_Ph=False)
            loss_P.backward()
            optimizer_Pu.step()

            # project

            with torch.no_grad():
                Du, Uu = torch.linalg.eigh(symmetric(Pu).detach().cpu())
                Du = Du.detach().cpu().numpy()
                Du_plus_diag = solve_constraint(Du, d=rank)
                Du = torch.tensor(np.diag(Du_plus_diag).real).float().to(device)
                Uu = Uu.to(device)
                Pu.data = Uu @ Du @ Uu.T
            
            count_examples += batch_size

        if i % evaluate_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            loss_val, score = get_score_param_free_twoPs(X_dev, U_dev, y_dev, Pu.detach().cpu(), Ph.detach().cpu(), rank, device)
            if loss_val > best_loss:#if np.abs(score - maj) < np.abs(best_score - maj):
                best_Pu, best_Ph, best_loss = symmetric(Pu).detach().cpu().numpy().copy(), symmetric(Ph).detach().cpu().numpy().copy(), loss_val
            if np.abs(score - maj) < np.abs(best_score - maj):
                best_score = score
                
            # update progress bar
            
            best_so_far = best_score if np.abs(best_score-maj) < np.abs(score-maj) else score
            
            pbar.set_description("{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(i, out_iters, score * 100, best_so_far * 100, maj * 100, np.abs(best_so_far - maj) * 100, best_loss, loss_val))
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        if i > 1 and np.abs(best_score - maj) < epsilon:
        #if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
                    break
    output = prepare_output_twoPs(best_Pu, best_Ph, rank, best_score, best_loss)
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
    
