import argparse
import epitran
import os
import pickle


epi_en = epitran.Epitran('eng-Latn')
epi_tl = epitran.Epitran('tgl-Latn')

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_vocab(path, lid_path):
    with open(path, 'r') as inf:
        lines = inf.readlines()
    with open(lid_path, 'r') as inf:
        lid_lines = inf.readlines()
    vocab = set()
    word_to_lid = {}
    for l, lid_l in zip(lines, lid_lines):
        l = l.strip()
        l_list = l.split()
        word_list = l_list[1:]
        lid_list = lid_l.strip().split()[1:]
        vocab = vocab.union(set(word_list))
        for word, lid in zip(word_list, lid_list):
            word_to_lid[word] = lid
    return vocab, word_to_lid

def mk_g2p_dict(vocab, word_to_lid):
    g2p_dict = {}
    for word in vocab:
        lid = word_to_lid[word]
        p = epi_en.transliterate(word) if lid == 'en' else epi_tl.transliterate(word)
        g2p_dict[word] = p
    return g2p_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train, dev, or test')
    return parser.parse_args()

def main():
    phase_file = '%s.txt' % args.phase
    lid_file = '%s_lids.txt' % args.phase
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    split_dir = os.path.join(parent_dir, 'split')
    phase_path = os.path.join(split_dir, phase_file)
    lid_path = os.path.join(split_dir, lid_file)

    vocab, word_to_lid = get_vocab(phase_path, lid_path)
    g2p_dict = mk_g2p_dict(vocab, word_to_lid)
    g2p_path = os.path.join(data_dir, 'g2p_dict_%s.pkl' % args.phase)
    save_pkl(g2p_dict, g2p_path)

if __name__ == '__main__':
    main()