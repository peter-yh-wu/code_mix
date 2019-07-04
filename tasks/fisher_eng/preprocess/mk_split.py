'''
Generates train-dev-test split

To-do:
 - Add conversation data to split

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
import random

def main():
    seed = 0
    train_frac = 0.8
    dev_frac = 0.1

    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    split_dir = os.path.join(parent_dir, 'split')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    text_dir = os.path.join(data_dir, 'text')

    train_path = os.path.join(split_dir, train_file)
    dev_path = os.path.join(split_dir, dev_file)
    test_path = os.path.join(split_dir, test_file)

    text_files = os.listdir(text_dir)
    text_files = [f for f in text_files if f.endswith('.txt')]
    text_paths = [os.path.join(text_dir, f) for f in text_files]

    all_lines = []
    for i, text_path in enumerate(text_paths):
        with open(text_path, 'r') as inf:
            curr_lines = inf.readlines()
        for l in curr_lines:
            l = l.strip()
            all_lines.append(l)

    random.seed(seed)
    random.shuffle(all_lines)

    num_samples = len(all_lines)
    num_train = int(train_frac*num_samples)
    num_dev = int(dev_frac*num_samples)

    train_lines = all_lines[:num_train]
    dev_lines = all_lines[num_train:num_train+num_dev]
    test_lines = all_lines[num_train+num_dev:]

    with open(train_path, 'w+') as ouf:
        for l in train_lines:
            ouf.write('%s\n' % l)

    with open(dev_path, 'w+') as ouf:
        for l in dev_lines:
            ouf.write('%s\n' % l)

    with open(test_path, 'w+') as ouf:
        for l in test_lines:
            ouf.write('%s\n' % l)

if __name__ == '__main__':
    main()