import itertools
import os

def rm_chars_in(path, lid_path):
    with open(path, 'r') as inf:
        lines = inf.readlines()
    with open(lid_path, 'r') as inf:
        lid_lines = inf.readlines()
    new_lines = []
    new_lid_lines = []
    for l, lid_l in zip(lines, lid_lines):
        l = l.strip()
        lid_l = lid_l.strip()
        l_list = l.split()
        lid_l_list = lid_l.split()
        new_l_list = []
        new_lid_l_list = []
        for _, (w, lid) in enumerate(zip(l_list, lid_l_list)):
            if w != '~' and w != '(())':
                w = w.replace('*', '')
                has_underscore = True if '_' in w else False
                w = w.replace('_', ' ')
                num_lids = len(w.split())
                if len(w) > 0:
                    new_l_list.append(w)
                    if has_underscore:
                        for _ in range(num_lids):
                            new_lid_l_list.append('en')
                    else:
                        new_lid_l_list.append(lid)
        if len(new_l_list) > 0:
            new_l = ' '.join(new_l_list)
            new_lid_l = ' '.join(new_lid_l_list)
            new_lines.append(new_l)
            new_lid_lines.append(new_lid_l)
    with open(path, 'w+') as ouf:
        for l in new_lines:
            ouf.write('%s\n' % l)
    with open(lid_path, 'w+') as ouf:
        for l in new_lid_lines:
            ouf.write('%s\n' % l)

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    split_dir = os.path.join(parent_dir, 'split')

    train_file = 'train.txt'
    dev_file = 'dev.txt'
    test_file = 'test.txt'

    train_path = os.path.join(split_dir, train_file)
    dev_path = os.path.join(split_dir, dev_file)
    test_path = os.path.join(split_dir, test_file)

    train_lid_file = 'train_lids.txt'
    dev_lid_file = 'dev_lids.txt'
    test_lid_file = 'test_lids.txt'

    train_lid_path = os.path.join(split_dir, train_lid_file)
    dev_lid_path = os.path.join(split_dir, dev_lid_file)
    test_lid_path = os.path.join(split_dir, test_lid_file)

    # rm_chars_in(train_path, train_lid_path)
    rm_chars_in(dev_path, dev_lid_path)
    rm_chars_in(test_path, test_lid_path)

if __name__ == '__main__':
    main()