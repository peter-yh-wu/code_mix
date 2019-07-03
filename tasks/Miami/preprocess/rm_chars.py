# -*- coding: utf-8 -*-
import os

bad_chars = ['ð', 'ʌ', 'λ', 'ʁ', '4', 'ɘ', 'ʣ', 'ɪ', 'ɛ', 'ɨ', 'ʘ', 'đ', \
    'ə', 'θ', 'ʍ', 'ǀ', 'ɾ', '3', 'ŋ', 'ʧ', 'ʃ', 'ɒ', 'ʤ', 'ʎ', 'ʊ']

def has_bad_char(l):
    l_list = l.split()
    words_list = l_list[1:]
    words = ' '.join(words_list)
    for ch in bad_chars:
        if ch in words:
            return True
    return False

def rm_bad_chars(lines, lid_lines):
    num_removed = 0
    new_lines = []
    new_lid_lines = []
    for l, lid_l in zip(lines, lid_lines):
        if not has_bad_char(l):
            new_lines.append(l.strip())
            new_lid_lines.append(lid_l.strip())
        else:
            num_removed += 1
    return new_lines, new_lid_lines, num_removed

def main():
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    split_dir = os.path.join(parent_dir, 'split')

    train_path = os.path.join(split_dir, train_file)
    dev_path = os.path.join(split_dir, dev_file)
    test_path = os.path.join(split_dir, test_file)

    train_lid_file = 'train_lids.txt'
    dev_lid_file = 'dev_lids.txt'
    test_lid_file = 'test_lids.txt'

    train_lid_path = os.path.join(split_dir, train_lid_file)
    dev_lid_path = os.path.join(split_dir, dev_lid_file)
    test_lid_path = os.path.join(split_dir, test_lid_file)

    with open(train_path, 'r') as inf:
        train_lines = inf.readlines()
    with open(train_lid_path, 'r') as inf:
        train_lid_lines = inf.readlines()
    with open(dev_path, 'r') as inf:
        dev_lines = inf.readlines()
    with open(dev_lid_path, 'r') as inf:
        dev_lid_lines = inf.readlines()
    with open(test_path, 'r') as inf:
        test_lines = inf.readlines()
    with open(test_lid_path, 'r') as inf:
        test_lid_lines = inf.readlines()

    train_lines, train_lid_lines, num_removed_train = rm_bad_chars(train_lines, train_lid_lines)
    dev_lines, dev_lid_lines, num_removed_dev = rm_bad_chars(dev_lines, dev_lid_lines)
    test_lines, test_lid_lines, num_removed_test = rm_bad_chars(test_lines, test_lid_lines)

    with open(train_path, 'w+') as ouf:
        for l in train_lines:
            ouf.write('%s\n' % l)

    with open(dev_path, 'w+') as ouf:
        for l in dev_lines:
            ouf.write('%s\n' % l)

    with open(test_path, 'w+') as ouf:
        for l in test_lines:
            ouf.write('%s\n' % l)

    with open(train_lid_path, 'w+') as ouf:
        for l in train_lid_lines:
            ouf.write('%s\n' % l)

    with open(dev_lid_path, 'w+') as ouf:
        for l in dev_lid_lines:
            ouf.write('%s\n' % l)

    with open(test_lid_path, 'w+') as ouf:
        for l in test_lid_lines:
            ouf.write('%s\n' % l)

    print(num_removed_train, num_removed_dev, num_removed_test)
    print(num_removed_train + num_removed_dev + num_removed_test)

if __name__ == '__main__':
    main()