import numpy as np
import pandas as pd

## utilities
import os

# Fix a bug: change the function name from evaluate_and_write to write_eval. (2020/01/20)
def write_eval(pred, label, save_dir, mode, epoch):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
  pred_df  = pd.DataFrame(pred)
  label_df = pd.DataFrame(label)
  pred_df.to_csv( os.path.join(save_dir, '%s_pred%d.csv'  % (mode, epoch)))
  label_df.to_csv(os.path.join(save_dir, '%s_label%d.csv' % (mode, epoch)))

  return { 'COR': COR(label, pred) }

def COR(label, pred):
  label_demeaned = label - label.mean(0)
  label_sumsquares = np.sum(np.square(label_demeaned), 0)
  pred_demeaned = pred - pred.mean(0)
  pred_sumsquares = np.sum(np.square(pred_demeaned), 0)
  cor_coef = np.diagonal(np.dot(label_demeaned.T, pred_demeaned)) / np.sqrt(label_sumsquares * pred_sumsquares)

  return np.nanmean(cor_coef)