import numpy as np
import os
import pickle

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

def p2g(p, p2g_dict, num_g):
    '''p is a string comprised of IPA characters'''
    gs = []
    while len(gs) < num_g:
        curr_i = 0
        while curr_i < len(p):
            pass
            # TODO
    return gs

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    g2p_path = os.path.join(data_dir, 'g2p_dict.pkl')
    g2p_dict = load_pkl(g2p_path)
    p2g_dict = {v: k for k, v in g2p_dict.items()}

    train_file = 'train.txt'
    split_dir = os.path.join(parent_dir, 'split')
    train_path = os.path.join(split_dir, train_file)
    
    # TODO
    keys = list(p2g_dict.keys())
    lens = [len(w) for w in keys]

    print(max(lens), sum(lens)/len(lens))
    distr = np.bincount(lens)/len(lens)
    print(np.random.multinomial(1, distr).where(array==item))

if __name__ == '__main__':
    main()