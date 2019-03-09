import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import csv
import numpy as np
import os
import pickle
import seaborn as sns
sns.set_palette('Set2')

from Levenshtein import distance

from model_utils import *

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

train_ids, dev_ids, test_ids = load_ids()
INTERVIEW_MFCC_DIR = '/home/srallaba/tools/kaldi/egs/seame/s5/feats_interview/cleaned'
test_xs, test_indices = load_x_data(test_ids, INTERVIEW_MFCC_DIR)
test_ys = load_y_data(test_indices, 'test') # 1-dim np array of strings

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
expt = 'lr_1e-3'
CSV_DIR = os.path.join(parent_dir, 'output', 'baseline', expt)

files = os.listdir(CSV_DIR)
files = [f for f in files if f.endswith('.csv')]
paths = [os.path.join(CSV_DIR, f) for f in files]

for i, p in enumerate(paths):
    dists = []
    norm_dists = []
    with open(p, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        for j, row in enumerate(raw_csv):
            y_pred = row[1]     # string
            y_true = test_ys[j] # string
            dist = distance(y_pred, y_true)
            dists.append(dist)
            norm_dist = dist / max(len(y_pred), len(y_true))
            norm_dists.append(norm_dist)
    metrics = {"dists": dists, "norm_dists": norm_dists}
    pkl_path = os.path.join(CSV_DIR, '%s.pkl' % files[i:-4])
    save_pkl(metrics, pkl_path)
    epochs = [(i+1)*4 for i, _ in enumerate(dists)]
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, dists)
    plt.title("Levenshtein Distance over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Levenshtein Distance")
    fig_path = os.path.join(CSV_DIR, '%s_lev.png' % files[i:-4])
    plt.savefig(fig_path)
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, norm_dists)
    plt.title("Levenshtein Distance over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Levenshtein Distance")
    fig_path = os.path.join(CSV_DIR, '%s_normlev.png' % files[i:-4])
    plt.savefig(fig_path)
