import os

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

    train_lid_file = 'train_lids.txt'
    dev_lid_file = 'dev_lids.txt'
    test_lid_file = 'test_lids.txt'

    train_lid_path = os.path.join(split_dir, train_lid_file)
    dev_lid_path = os.path.join(split_dir, dev_lid_file)
    test_lid_path = os.path.join(split_dir, test_lid_file)

    lid_dir = os.path.join(data_dir, 'lids')
    lid_files = os.listdir(lid_dir)
    lid_files = [f for f in lid_files if f.endswith('.txt')]
    lid_paths = [os.path.join(lid_dir, f) for f in lid_files]

    lid_dict = {}
    for i, lid_path in enumerate(lid_paths):
        with open(lid_path, 'r') as inf:
            curr_lines = inf.readlines()
        for l in curr_lines:
            l = l.strip()
            l_list = l.split()
            fid = l_list[0]
            lid_dict[fid] = l

    train_lid_lines = []
    dev_lid_lines = []
    test_lid_lines = []
    with open(train_path, 'r') as inf:
        train_lines = inf.readlines()
    for l in train_lines:
        fid = l.strip().split()[0]
        train_lid_lines.append(lid_dict[fid])
    with open(dev_path, 'r') as inf:
        dev_lines = inf.readlines()
    for l in dev_lines:
        fid = l.strip().split()[0]
        dev_lid_lines.append(lid_dict[fid])
    with open(test_path, 'r') as inf:
        test_lines = inf.readlines()
    for l in test_lines:
        fid = l.strip().split()[0]
        test_lid_lines.append(lid_dict[fid])

    with open(train_lid_path, 'w+') as ouf:
        for l in train_lid_lines:
            ouf.write('%s\n' % l)

    with open(dev_lid_path, 'w+') as ouf:
        for l in dev_lid_lines:
            ouf.write('%s\n' % l)

    with open(test_lid_path, 'w+') as ouf:
        for l in test_lid_lines:
            ouf.write('%s\n' % l)

if __name__ == '__main__':
    main()