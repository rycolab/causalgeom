#%%
import warnings
import logging
import os
import sys
import coloredlogs

import numpy as np
import torch
from sklearn.metrics import log_loss

from classifiers.classifiers import BinaryParamFreeClf, BinaryParamFreeClfTwoPs
from rlace import init_classifier, get_majority_acc

#%% Diagnostic probe performance before and after
def eval_diagnostic(P, I_P, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    
    svm = init_classifier()
    svm.fit(X_train, y_train)
    results["diag_acc_original"] = svm.score(X_test, y_test)
    results["diag_loss_original"] = log_loss(
        y_test, svm.predict_proba(X_test)
    )

    svm2 = init_classifier()
    svm2.fit(X_train @ P , y_train)
    results["diag_acc_P_train"] = svm2.score(X_train @ P, y_train)
    results["diag_acc_P_val"] = svm2.score(X_val @ P, y_val)
    results["diag_acc_P_test"] = svm2.score(X_test @ P, y_test)
    results["diag_loss_P_test"] = log_loss(
        y_test, svm2.predict_proba(X_test @ P)
    )
    results["diag_loss_P_train"] = log_loss(
        y_train, svm.predict_proba(X_train @ P)
    )

    svm3 = init_classifier()
    svm3.fit(X_train @ I_P , y_train)
    results["diag_acc_I_P_train"] = svm3.score(X_train @ I_P, y_train)
    results["diag_acc_I_P_val"] = svm3.score(X_val @ I_P, y_val)
    results["diag_acc_I_P_test"] = svm3.score(X_test @ I_P, y_test)
    results["diag_loss_I_P_test"] = log_loss(
        y_test, svm3.predict_proba(X_test @ I_P)
    )
    results["diag_loss_I_P_train"] = log_loss(
        y_train, svm.predict_proba(X_train @ I_P)
    )
    return results

def eval_usage(P, I_P, X_train, U_train, y_train, X_test, U_test, y_test, Pu = None):
    results = {}
    
    if Pu is not None:
        trainclf = BinaryParamFreeClfTwoPs(X_train, U_train, y_train, Pu, P, "cpu")
        #disag_trainclf = BinaryParamFreeClfTwoPs(X_train, U_train, 1-y_train, Pu, P, "cpu")
        testclf = BinaryParamFreeClfTwoPs(X_test, U_test, y_test, Pu, P, "cpu")
        #disag_testclf = BinaryParamFreeClfTwoPs(X_test, U_test, 1-y_test, Pu, P, "cpu")
    else:
        trainclf = BinaryParamFreeClf(X_train, U_train, y_train, P, I_P, "cpu")
        #disag_trainclf = BinaryParamFreeClf(X_train, U_train, 1-y_train, P, "cpu")
        testclf = BinaryParamFreeClf(X_test, U_test, y_test, P, I_P, "cpu")
        #disag_testclf = BinaryParamFreeClf(X_test, U_test, 1-y_test, P, "cpu")
    
    results["lm_acc_original"] = testclf.score()
    results["lm_acc_P_test"] = testclf.score_P()
    results["lm_acc_I_P_test"] = testclf.score_I_P()

    results["lm_loss_original"] = testclf.loss()
    results["lm_loss_P_test"] = testclf.loss_P()
    results["lm_loss_I_P_test"] = testclf.loss_I_P()

    #results["lm_loss_original_disag"] = disag_testclf.loss()
    #results["lm_loss_P_test_disag"] = disag_testclf.loss_P()
    #results["lm_loss_I_P_test_disag"] = disag_testclf.loss_I_P()

    results["lm_acc_P_train"] = trainclf.score_P()
    results["lm_acc_I_P_train"] = trainclf.score_I_P()
    results["lm_loss_P_train"] = trainclf.loss_P()
    results["lm_loss_I_P_train"] = trainclf.loss_I_P()

    #results["lm_loss_projected_train_disag"] = disag_trainclf.loss_P()
    #results["lm_loss_I_P_train_disag"] = disag_trainclf.loss_I_P()
    return results

def full_usage_eval(output, X_train, U_train, y_train, X_val, U_val, y_val, X_test, U_test, y_test, runtime):
    results = {}
    
    P = output["P"]
    I_P = output["I_P"]
    P_acc = output["P_acc"]
    I_P_acc = output["I_P_acc"]
    try:
        Pu = output["Pu"]
    except KeyError:
        Pu = None
    
    try:
        val_results = output["val_results"]
    except KeyError:
        val_results = None
        
    results["P"] = P
    results["P_acc"] = P_acc
    results["I_P"] = I_P
    results["I_P_acc"] = I_P_acc
    results["Pu"] = Pu
    results["val_results"] = val_results
    results["runtime"] = runtime
    results["optim_best_acc"] = output["best_acc"]
    results["optim_best_loss"] = output.get("best_loss", None)
    
    results_diag = eval_diagnostic(P, I_P, X_train, y_train, X_val, y_val, X_test, y_test)
    results_usage = eval_usage(P, I_P, X_train, U_train, y_train, X_test, U_test, y_test, Pu)
    return results | results_diag | results_usage
