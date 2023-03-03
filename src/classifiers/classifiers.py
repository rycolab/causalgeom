import numpy as np
import pickle
import torch
import sklearn

#%%
class BinaryParamFreeClf():
    def __init__(self, X, U, y, device):
        self.X = torch.tensor(X).float()
        self.U = torch.tensor(U).float()
        self.y = y

        self.clf = torch.nn.Sigmoid()

    def get_logits(self):
        return (self.X * self.U).sum(-1)

    def predict_proba(self):
        return self.clf(self.get_logits()).cpu().numpy()

    def __get_class_from_probas(self, probas):
        return (probas>0.5).astype(float)

    def predict(self):
        return self.__get_class_from_probas(self.predict_proba())

    def loss(self):
        return sklearn.metrics.log_loss(self.y, self.predict_proba())
            
    def __compute_acc(self, y_pred):
        """ y_pred has to be {0, 1}"""
        return 1 - (np.sum(np.not_equal(y_pred, self.y)) / self.y.shape[0])

    def score(self):
        return self.__compute_acc(self.predict())

#%% 
class BinaryParamFreeClfPs():
    def __init__(self, X, U, y, P, I_P, device):
        self.X = torch.tensor(X).float()
        self.U = torch.tensor(U).float()
        self.y = y
        self.P = torch.tensor(P).float()
        self.I_P = torch.tensor(I_P).float()
        #self.eye = torch.eye(self.X.shape[1]).to(device)

        #self.compP = self.eye - self.P
        self.clf = torch.nn.Sigmoid()

    def get_logits(self):
        return (self.X * self.U).sum(-1)

    def get_logits_P(self):
        return ((self.X @ self.P) * self.U).sum(-1)

    def get_logits_I_P(self):
        return ((self.X @ self.I_P) * self.U).sum(-1)
    
    def predict_proba(self):
        return self.clf(self.get_logits()).cpu().numpy()

    def predict_P_proba(self):
        return self.clf(self.get_logits_P()).cpu().numpy()
    
    def predict_I_P_proba(self):
        return self.clf(self.get_logits_I_P()).cpu().numpy()

    def __get_class_from_probas(self, probas):
        return (probas>0.5).astype(float)

    def predict(self):
        return self.__get_class_from_probas(self.predict_proba())

    def predict_P(self):
        return self.__get_class_from_probas(self.predict_P_proba())
    
    def predict_I_P(self):
        return self.__get_class_from_probas(self.predict_I_P_proba())
    
    def loss(self):
        return sklearn.metrics.log_loss(self.y, self.predict_proba())
    
    def loss_P(self):
        return sklearn.metrics.log_loss(self.y, self.predict_P_proba())
    
    def loss_I_P(self):
        return sklearn.metrics.log_loss(self.y, self.predict_I_P_proba())
        
    def __compute_acc(self, y_pred):
        """ y_pred has to be {0, 1}"""
        return 1 - (np.sum(np.not_equal(y_pred, self.y)) / self.y.shape[0])

    def score(self):
        return self.__compute_acc(self.predict())

    def score_P(self):
        return self.__compute_acc(self.predict_P())
    
    def score_I_P(self):
        return self.__compute_acc(self.predict_I_P())

#%%
class BinaryParamFreeClfTwoPs():
    def __init__(self, X, U, y, Pu, Ph, device):
        self.X = torch.tensor(X).float()
        self.U = torch.tensor(U).float()
        self.y = y
        self.Pu = torch.tensor(Pu).float()
        self.Ph = torch.tensor(Ph).float()
        self.eye = torch.eye(self.X.shape[1]).to(device)

        self.compPu = self.eye - self.Pu
        self.compPh = self.eye - self.Ph

        self.clf = torch.nn.Sigmoid()

    def get_logits(self):
        return (self.X * self.U).sum(-1)

    def get_logits_P(self):
        return ((self.X @ self.Ph) * (self.U @ self.Pu)).sum(-1)

    def get_logits_compP(self):
        return ((self.X @ self.compPh) * (self.U @ self.compPu)).sum(-1)
    
    def predict_proba(self):
        return self.clf(self.get_logits()).cpu().numpy()

    def predict_P_proba(self):
        return self.clf(self.get_logits_P()).cpu().numpy()
    
    def predict_compP_proba(self):
        return self.clf(self.get_logits_compP()).cpu().numpy()

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
