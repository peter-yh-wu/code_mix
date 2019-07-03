import os

def lowercase(path):
    with open(path, 'r') as inf:
        lines = inf.readlines()
    new_lines = []
    for l in lines:
        l = l.strip()
        l_list = l.split()
        fid = l_list[0]
        words = ' '.join(l_list[1:])
        new_words = words.lower()
        new_l = fid + ' ' + new_words
        new_lines.append(new_l)
    with open(path, 'w+') as ouf:
        for l in new_lines:
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

    lowercase(train_path)
    lowercase(dev_path)
    lowercase(test_path)

if __name__ == '__main__':
    main()