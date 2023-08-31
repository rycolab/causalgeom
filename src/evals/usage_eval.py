#%%
import warnings
import logging
import os
import sys
import coloredlogs

import numpy as np
import torch
from sklearn.metrics import log_loss

#sys.path.append('..')
sys.path.append('./src/')

from classifiers.classifiers import BinaryParamFreeClf, BinaryParamFreeClfTwoPs
from algorithms.rlace.rlace import init_classifier, get_majority_acc

#%% Diagnostic probe performance before and after
def diag_eval(P, P_type, X_train, y_train, X_val, y_val, X_test, y_test, X_pca=None):
    results = {}
    svm = init_classifier()

    if X_pca is None:
        X_train_proj = X_train @ P
        X_val_proj = X_val @ P
        X_test_proj = X_test @ P
    else:
        X_train_proj = X_pca.inverse_transform(X_pca.transform(X_train) @ P)
        X_val_proj = X_pca.inverse_transform(X_pca.transform(X_val) @ P)
        X_test_proj = X_pca.inverse_transform(X_pca.transform(X_test) @ P)
    
    svm.fit(X_train_proj, y_train)

    results[f"diag_acc_{P_type}_train"] = svm.score(X_train_proj, y_train)
    results[f"diag_acc_{P_type}_val"] = svm.score(X_val_proj, y_val)
    results[f"diag_acc_{P_type}_test"] = svm.score(X_test_proj, y_test)
    
    results[f"diag_loss_{P_type}_train"] = log_loss(
        y_train, svm.predict_proba(X_train_proj)
    )
    results[f"diag_loss_{P_type}_val"] = log_loss(
        y_val, svm.predict_proba(X_val_proj)
    )
    results[f"diag_loss_{P_type}_test"] = log_loss(
        y_test, svm.predict_proba(X_test_proj)
    )
    return results

def full_diag_eval(P, I_P, X_train, y_train, X_val, y_val, X_test, y_test, X_pca=None):
    I = np.eye(P.shape[1], P.shape[1])

    diag_orig = diag_eval(
        I, "original", X_train, y_train, X_val, y_val, X_test, y_test
    )
    #diag_P = diag_eval(P, "P", X_train, y_train, X_val, y_val, X_test, y_test, X_pca=X_pca)
    #diag_I_P = diag_eval(I_P, "I_P", X_train, y_train, X_val, y_val, X_test, y_test, X_pca=X_pca)

    #diag_P_acc = diag_eval(P_acc, "P_acc", X_train, y_train, X_val, y_val, X_test, y_test, X_pca=X_pca)
    #diag_I_P_acc = diag_eval(I_P_acc, "I_P_acc", X_train, y_train, X_val, y_val, X_test, y_test, X_pca=X_pca)

    diag_P = diag_eval(
        P, "P", X_train, y_train, X_val, y_val, X_test, y_test
    )
    diag_I_P = diag_eval(
        I_P, "I_P", X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    return diag_orig | diag_P | diag_I_P #| diag_P | diag_I_P | diag_P_acc | diag_I_P_acc | 

def usage_eval(P, P_type, X_train, U_train, y_train, 
    X_test, U_test, y_test, X_pca=None):
    results = {}
    
    if X_pca is None:
        X_train_proj = X_train @ P
        X_test_proj = X_test @ P
    else:
        X_train_proj = X_pca.inverse_transform(X_pca.transform(X_train) @ P)
        X_test_proj = X_pca.inverse_transform(X_pca.transform(X_test) @ P)
        
    train_clf = BinaryParamFreeClf(X_train_proj, U_train, y_train, "cpu")
    test_clf = BinaryParamFreeClf(X_test_proj, U_test, y_test, "cpu")

    results[f"lm_acc_{P_type}_train"] = train_clf.score()
    results[f"lm_loss_{P_type}_train"] = train_clf.loss()

    results[f"lm_acc_{P_type}_test"] = test_clf.score()
    results[f"lm_loss_{P_type}_test"] = test_clf.loss()

    return results

def full_usage_eval(P, I_P, X_train, U_train, y_train, X_test, U_test, y_test, X_pca=None):

    usage_orig = usage_eval(
        np.eye(P.shape[0], P.shape[1]), "original", 
        X_train, U_train, y_train, X_test, U_test, y_test
    )
    #usage_P = usage_eval(P, "P", X_train, U_train, y_train, X_test, U_test, y_test, X_pca)
    #usage_I_P = usage_eval(I_P, "I_P", X_train, U_train, y_train, X_test, U_test, y_test, X_pca)

    #usage_P_acc = usage_eval(P_acc, "P_acc", X_train, U_train, y_train, X_test, U_test, y_test, X_pca)
    #usage_I_P_acc = usage_eval(I_P_acc, "I_P_acc", X_train, U_train, y_train, X_test, U_test, y_test, X_pca)

    usage_P = usage_eval(
        P, "P", X_train, U_train, y_train, X_test, U_test, y_test
    )
    usage_I_P = usage_eval(
        I_P, "I_P", X_train, U_train, y_train, X_test, U_test, y_test
    )

    return usage_orig | usage_P | usage_I_P #| usage_P | usage_I_P | usage_P_acc | usage_I_P_acc 
