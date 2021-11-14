import numpy as np
import torch

class EarlyStopping:
    """
    EarlyStopping taken from avishekthakur/wtfml
    """
    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
        
        
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        



import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_dataframe(csv, folds=None, frac=1.0):
    df = pd.read_csv(csv, keep_default_na=False).sample(frac=frac, random_state=123).reset_index(drop=True)
    
    if folds is not None:
        df['fold'] = -1
        
        if folds > 0:
            y = df.label.values
            kf = StratifiedKFold(n_splits=folds)

            for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
                df.loc[v_, 'fold'] = f
        
    return df