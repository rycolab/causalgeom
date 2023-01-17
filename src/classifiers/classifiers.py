import numpy as np
import pickle
import torch
import sklearn

#%% 
class BinaryParamFreeClf():
    def __init__(self, X, U, y, P):
        self.X = torch.from_numpy(X).float()
        self.U = torch.from_numpy(U).float()
        self.y = y
        self.P = torch.from_numpy(P).float()

        self.compP = torch.eye(X.shape[1]) - self.P
        self.clf = torch.nn.Sigmoid()

        self.logits = torch.sum(
            torch.mm(self.X, torch.t(self.U)), dim = 1
        )
        self.logits_P = torch.sum(
            torch.mm(self.X @ self.P, torch.t(self.U)), dim = 1
        )
        self.logits_compP = torch.sum(
            torch.mm(self.X @ (self.compP), torch.t(self.U)), dim = 1
        )

    def get_logits(self):
        return self.logits

    def get_logits_P(self):
        return self.logits_P

    def get_logits_compP(self):
        return self.logits_compP
    
    def predict_proba(self):
        return self.clf(self.logits).numpy()

    def predict_P_proba(self):
        return self.clf(self.logits_P).numpy()
    
    def predict_compP_proba(self):
        return self.clf(self.logits_compP).numpy()

    def __get_class_from_probas(self, probas):
        return (probas>0.5).astype(float)

    def predict(self):
        return self.__get_class_from_probas(self.predict_proba())

    def predict_P(self):
        return self.__get_class_from_probas(self.predict_P_proba())
    
    def predict_compP(self):
        return self.__get_class_from_probas(self.predict_compP_proba())
    
    def loss(self):
        return sklearn.metrics.log_loss(self.y, self.predict_proba())
    
    def loss_P(self):
        return sklearn.metrics.log_loss(self.y, self.predict_P_proba())
    
    def loss_compP(self):
        return sklearn.metrics.log_loss(self.y, self.predict_compP_proba())
        
    def __compute_acc(self, y_pred):
        """ y_pred has to be {0, 1}"""
        return 1 - (np.sum(np.not_equal(y_pred, self.y)) / self.y.shape[0])

    def score(self):
        return self.__compute_acc(self.predict())

    def score_P(self):
        return self.__compute_acc(self.predict_P())
    
    def score_compP(self):
        return self.__compute_acc(self.predict_compP())
