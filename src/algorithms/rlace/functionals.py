import torch

class PositivelyFunctional:
    def __init__(self, get_loss_fn, get_score_param_free):
        self.best_loss = float('inf')
        self.best_acc = 0
        self.get_loss_fn = get_loss_fn
        self.get_score_param_free = get_score_param_free

    def get_loss(self, X_batch, U_batch, y_batch, P, bce_loss_fn, device):
        I = torch.eye(P.shape[0]).to(device)
        return self.get_loss_fn(
            X_batch, U_batch, y_batch, 
            I - P, bce_loss_fn, optimize_P=False
        )

    def is_best_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def is_best_acc(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            return True
        return False

    def get_loss_and_score(self, X_dev, U_dev, y_dev, P, rank, device):
        I = torch.eye(P.shape[0]).to(device)
        return self.get_score_param_free(
            X_dev, U_dev, y_dev, (I - P).detach().cpu(), rank, device)


class NegativelyFunctional:
    def __init__(self, get_loss_fn, get_score_param_free):
        self.best_loss = 0
        self.best_acc = float('inf')
        self.get_loss_fn = get_loss_fn
        self.get_score_param_free = get_score_param_free

    def get_loss(self, X_batch, U_batch, y_batch, P, bce_loss_fn, device):
        return self.get_loss_fn(X_batch, U_batch, 1 - y_batch,
                           P, bce_loss_fn, optimize_P=False)

    def is_best_loss(self, loss):
        if loss > self.best_loss:
            self.best_loss = loss
            return True
        return False

    def is_best_acc(self, acc):
        if acc < self.best_acc:
            self.best_acc = acc
            return True
        return False

    def get_loss_and_score(self, X_dev, U_dev, y_dev, P, rank, device):
        return self.get_score_param_free(
            X_dev, U_dev, 1 - y_dev, P.detach().cpu(), rank, device)


class OriginalFunctional:
    def __init__(self, get_loss_fn, get_score_param_free):
        self.best_loss = 0
        self.best_acc = float('inf')
        self.get_loss_fn = get_loss_fn
        self.get_score_param_free = get_score_param_free

    def get_loss(self, X_batch, U_batch, y_batch, P, bce_loss_fn, device):
        return self.get_loss_fn(X_batch, U_batch, y_batch,
                           P, bce_loss_fn, optimize_P=True)

    def is_best_loss(self, loss):
        if loss > self.best_loss:
            self.best_loss = loss
            return True
        return False

    def is_best_acc(self, acc):
        if acc < self.best_acc:
            self.best_acc = acc
            return True
        return False

    def get_loss_and_score(self, X_dev, U_dev, y_dev, P, rank, device):
        return self.get_score_param_free(
            X_dev, U_dev, y_dev, P.detach().cpu(), rank, device)
