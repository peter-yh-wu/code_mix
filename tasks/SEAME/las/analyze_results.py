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

def mk_loss_curves():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expt = 'lr_1e-3'
    CSV_DIR = os.path.join(parent_dir, 'output', 'baseline', expt)
    train_loss_path = os.path.join(CSV_DIR, 'train_losses.txt')
    val_loss_path = os.path.join(CSV_DIR, 'val_losses.txt')
    with open(train_loss_path, 'r') as inf:
        lines = inf.readlines()
    train_losses = [float(l.strip()) for l in lines]
    with open(val_loss_path, 'r') as inf:
        lines = inf.readlines()
    val_losses = [float(l.strip()) for l in lines]
    plt.figure(figsize=(8, 8))
    epochs = [(e+1) for e in range(len(train_losses))]
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc='upper right', fontsize='large')
    fig_path = os.path.join(CSV_DIR, 'loss_curves.png')
    plt.savefig(fig_path)

def plt_lev_by_id():
    _, _, test_ids = load_ids()
    INTERVIEW_MFCC_DIR = '/home/srallaba/tools/kaldi/egs/seame/s5/feats_interview/cleaned'
    _, test_indices = load_x_data(test_ids, INTERVIEW_MFCC_DIR)
    test_ys = load_y_data(test_indices, 'test') # len-4063 np array of strings

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
            for j, row in enumerate(raw_csv): # row is size-2 list
                y_pred = row[1]     # string
                y_true = test_ys[j] # string
                dist = distance(y_pred, y_true)
                dists.append(dist)
                norm_dist = dist / max(len(y_pred), len(y_true), 1)
                norm_dists.append(norm_dist)
        metrics = {"dists": dists, "norm_dists": norm_dists}
        pkl_path = os.path.join(CSV_DIR, '%s.pkl' % files[i][:-4])
        save_pkl(metrics, pkl_path)
        fids = [(e+1) for e, _ in enumerate(dists)]
        plt.figure(figsize=(10, 10))
        plt.plot(fids, dists)
        plt.title("Levenshtein Distance per Sample")
        plt.xlabel("Sample ID")
        plt.ylabel("Levenshtein Distance")
        fig_path = os.path.join(CSV_DIR, '%s_lev.png' % files[i][:-4])
        plt.savefig(fig_path)
        plt.figure(figsize=(10, 10))
        plt.plot(fids, norm_dists)
        plt.title("Normalized Levenshtein Distance per Sample")
        plt.xlabel("Sample ID")
        plt.ylabel("Normalized Levenshtein Distance")
        fig_path = os.path.join(CSV_DIR, '%s_normlev.png' % files[i][:-4])
        plt.savefig(fig_path)

def plt_lev_by_epoch():
    _, _, test_ids = load_ids()
    INTERVIEW_MFCC_DIR = '/home/srallaba/tools/kaldi/egs/seame/s5/feats_interview/cleaned'
    _, test_indices = load_x_data(test_ids, INTERVIEW_MFCC_DIR)
    test_ys = load_y_data(test_indices, 'test') # len-4063 np array of strings

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expt = 'lr_1e-3'
    CSV_DIR = os.path.join(parent_dir, 'output', 'baseline', expt)

    files = os.listdir(CSV_DIR)
    files = [f for f in files if f.endswith('.csv')]
    paths = [os.path.join(CSV_DIR, f) for f in files]

    epochs = [(e+1)*4 for e, _ in enumerate(paths)]

    dist_means = []
    dist_vars = []
    norm_dist_means = []
    norm_dist_vars = []
    for i, p in enumerate(paths):
        dists = []
        norm_dists = []
        with open(p, 'r') as csvfile:
            raw_csv = csv.reader(csvfile)
            for j, row in enumerate(raw_csv): # row is size-2 list
                y_pred = row[1]     # string
                y_true = test_ys[j] # string
                dist = distance(y_pred, y_true)
                dists.append(dist)
                norm_dist = dist / max(len(y_pred), len(y_true), 1)
                norm_dists.append(norm_dist)
        dist_mean = np.mean(dists)
        dist_var = np.var(dists)
        norm_dist_mean = np.mean(norm_dists)
        norm_dist_var = np.var(norm_dists)
        dist_means.append(dist_mean)
        dist_vars.append(dist_var)
        norm_dist_means.append(norm_dist_mean)
        norm_dist_vars.append(norm_dist_var)

    plt.figure(figsize=(10, 10))
    # plt.errorbar(epochs, dist_means, dist_vars)
    plt.plot(epochs, dist_means)
    plt.title("Levenshtein Distance over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Levenshtein Distance")
    fig_path = os.path.join(CSV_DIR, 'lev.png')
    plt.savefig(fig_path)
    plt.figure(figsize=(10, 10))
    # plt.errorbar(epochs, norm_dist_means, norm_dist_vars)
    plt.plot(epochs, norm_dist_means)
    plt.title("Normalized Levenshtein Distance over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Levenshtein Distance")
    fig_path = os.path.join(CSV_DIR, 'normlev.png')
    plt.savefig(fig_path)

    metrics = {
        "dist_means": dist_means, 
        "dist_vars": dist_vars,
        "norm_dist_means": norm_dist_means,
        "norm_dist_vars": norm_dist_vars
    }
    pkl_path = os.path.join(CSV_DIR, 'metrics.pkl')
    save_pkl(metrics, pkl_path)

def main():
    mk_loss_curves()

if __name__ == '__main__':
    main()