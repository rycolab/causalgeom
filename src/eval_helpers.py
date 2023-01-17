#%%
import warnings
import logging
import os
import sys
import coloredlogs

import numpy as np
import torch
from sklearn.metrics import log_loss

from classifiers.classifiers import BinaryParamFreeClf
from rlace import init_classifier, get_majority_acc

#%% Diagnostic probe performance before and after
def eval_diagnostic(P, X_train, y_train, X_test, y_test):
    results = {}
    
    svm = init_classifier()
    svm.fit(X_train[:] , y_train[:])
    results["diag_acc_original"] = svm.score(X_test, y_test)
    results["diag_loss_original"] = log_loss(
        y_test, svm.predict_proba(X_test)
    )

    svm = init_classifier()
    svm.fit(X_train[:] @ P , y_train[:])
    results["diag_acc_projected_test"] = svm.score(X_test @ P, y_test)
    results["diag_acc_projected_train"] = svm.score(X_train @ P, y_train)
    results["diag_loss_projected_test"] = log_loss(
        y_test, svm.predict_proba(X_test @ P)
    )
    results["diag_loss_projected_train"] = log_loss(
        y_train, svm.predict_proba(X_train @ P)
    )

    compP = np.eye(X_train.shape[1]) - P
    svm = init_classifier()
    svm.fit(X_train[:] @ compP , y_train[:])
    results["diag_acc_comp_projected_train"] = svm.score(X_train @ compP, y_train)
    results["diag_acc_comp_projected_test"] = svm.score(X_test @ compP, y_test)
    results["diag_loss_comp_projected_test"] = log_loss(
        y_test, svm.predict_proba(X_test @ compP)
    )
    results["diag_loss_comp_projected_train"] = log_loss(
        y_train, svm.predict_proba(X_train @ compP)
    )
    return results

def eval_usage(P, X_train, U_train, y_train, X_test, U_test, y_test):
    results = {}
    
    trainclf = BinaryParamFreeClf(X_train, U_train, y_train, P)
    testclf = BinaryParamFreeClf(X_test, U_test, y_test, P)
    
    results["lm_acc_original"] = testclf.score()
    results["lm_acc_projected_test"] = testclf.score_P()
    results["lm_acc_comp_projected_test"] = testclf.score_compP()

    results["lm_loss_original"] = testclf.loss()
    results["lm_loss_projected_test"] = testclf.loss_P()
    results["lm_loss_comp_projected_test"] = testclf.loss_compP()
    
    results["lm_acc_projected_train"] = trainclf.score_P()
    results["lm_acc_comp_projected_train"] = trainclf.score_compP()
    results["lm_loss_projected_train"] = trainclf.loss_P()
    results["lm_loss_comp_projected_train"] = trainclf.loss_compP()
    return results

def full_eval(output, X_train, U_train, y_train, X_test, U_test, y_test, runtime):
    P = output["P"]
    
    results = {}

    results["P"] = P
    results["runtime"] = runtime
    results["optim_best_acc"] = output["best_score"]
    results["optim_best_loss"] = output.get("best_loss", None)
    
    results_diag = eval_diagnostic(P, X_train, y_train, X_test, y_test)
    results_usage = eval_usage(P, X_train, U_train, y_train, X_test, U_test, y_test)
    return results | results_diag | results_usage
