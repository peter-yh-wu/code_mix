import argparse
import numpy as np
import os
import pickle
import sys

from nltk.metrics import edit_distance


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def g2p(g, g2p_dict):
    '''g is a string of words'''
    l_list = g.split()
    p_list = []
    for w in l_list:
        p_list.append(g2p_dict[w])
    p = ' '.join(p_list)
    return p


def dist(p1, p2):
    return edit_distance(p1, p2)


def find_closest_word_helper(p, p2g_dict_by_len):
    len_p = len(p)
    min_dist_1 = sys.maxsize
    best_p_1 = ''
    for curr_p in p2g_dict_by_len[len_p]:
        curr_dist = dist(p, curr_p)
        if curr_dist < min_dist_1:
            min_dist_1 = curr_dist
            best_p_1 = curr_p
    
    min_dist_2 = sys.maxsize
    if len_p > 2:
        best_p_2 = ''
        for curr_p in p2g_dict_by_len[len_p-1]:
            curr_dist = dist(p, curr_p)
            if curr_dist < min_dist_2:
                min_dist_2 = curr_dist
                best_p_2 = curr_p

    min_dist_3 = sys.maxsize
    if len_p > 4:
        best_p_3 = ''
        for curr_p in p2g_dict_by_len[len_p-2]:
            curr_dist = dist(p, curr_p)
            if curr_dist < min_dist_3:
                min_dist_3 = curr_dist
                best_p_3 = curr_p

    best_p = best_p_1 if min_dist_1 < min_dist_2 and min_dist_1 < min_dist_3 else best_p_2 if min_dist_2 < min_dist_3 else best_p_3
    return best_p, p2g_dict_by_len[len(best_p)][best_p]


def find_closest_word(p, p2g_dict_by_len):
    new_p = p
    len_p = len(new_p)
    if new_p in p2g_dict_by_len[len_p]:
        return new_p, p2g_dict_by_len[len_p][new_p]
    elif len_p > 2 and new_p[:-1] in p2g_dict_by_len[len_p-1]:
        return new_p[:-1], p2g_dict_by_len[len_p-1][new_p[:-1]]
    elif len_p > 4 and new_p[:-2] in p2g_dict_by_len[len_p-2]:
        return new_p[:-2], p2g_dict_by_len[len_p-2][new_p[:-2]]
    else:
        return find_closest_word_helper(p, p2g_dict_by_len)


def p2g(p, p2g_dict_by_len, distr, num_g):
    '''p is a string comprised of IPA characters'''
    gs = []
    while len(gs) < num_g:
        g_sent_list = []
        curr_i = 0
        while curr_i < len(p):
            curr_len = np.where(np.random.multinomial(1, distr)==1)[0][0]
            remaining_len = len(p)-curr_i
            if remaining_len < curr_len:
                curr_len = remaining_len
            curr_p = p[curr_i:curr_i+curr_len]
            best_p, g_word = find_closest_word(curr_p, p2g_dict_by_len)
            curr_i += curr_len # len(best_p)
            g_sent_list.append(g_word)
        g_sent = ' '.join(g_sent_list)
        gs.append(g_sent)
    return gs

def mk_gs(g, g2p_dict, p2g_dict_by_len, distr, num_g):
    p = g2p(g, g2p_dict)
    gs = p2g(p, p2g_dict_by_len, distr, num_g)
    return gs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-g', type=int, default=5, metavar='N', help='number of new sentences per datapoint')
    return parser.parse_args()

def main():
    args = parse_args()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    g2p_path = os.path.join(data_dir, 'g2p_dict.pkl')
    g2p_dict = load_pkl(g2p_path)
    p2g_dict = {v: k for k, v in g2p_dict.items()}

    train_file = 'train.txt'
    split_dir = os.path.join(parent_dir, 'split')
    train_path = os.path.join(split_dir, train_file)
    
    ps = list(p2g_dict.keys())
    lens = [len(w) for w in ps]
    max_len = max(lens)
    distr = np.bincount(lens)/len(lens)
    p2g_dict_by_len = [{}]*(max_len+1)
    for p in p2g_dict:
        p2g_dict_by_len[len(p)][p] = p2g_dict[p]

    with open(train_path, 'r') as inf:
        lines = inf.readlines()

    all_gs = []
    for l in lines:
        l = l.strip()
        l_list = l.split()
        fid = l_list[0]
        print(fid)
        words = ' '.join(l_list[1:])
        num_raw_gs = args.num_g*2
        gs = mk_gs(words, g2p_dict, p2g_dict_by_len, distr, num_raw_gs)
        dists = [edit_distance(words, g) for g in gs]
        idxs = np.argsort(dists)
        sorted_gs = np.array(gs)[idxs]
        best_gs = sorted_gs[:args.num_g]
        best_gs = np.insert(best_gs, 0, fid)
        for g in best_gs:
            print(g)
        all_gs.append(best_gs)
    all_gs = np.stack(all_gs)
    gs_path = os.path.join(data_dir, 'gs.csv')
    np.savetxt(gs_path, all_gs, delimiter=",")


if __name__ == '__main__':
    main()