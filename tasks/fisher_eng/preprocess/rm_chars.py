import os

def no_number_in(words):
    digits = [0,1,2,3,4,5,6,7,8,9]
    for d in digits:
        if str(d) in words:
            return False
    return True


def process_txt(path):
    with open(path, 'r') as inf:
        lines = inf.readlines()
    new_lines = []
    for l in lines:
        l = l.strip()
        l_list = l.split()
        fid = l_list[0]
        words_list = l_list[1:]
        words = ' '.join(words_list)
        if '<' not in words and no_number_in(words):
            words = words.replace('?', '')
            words = words.replace('.', '')
            words = words.replace(',', '')
            words = words.replace('*', '')
            words = words.replace("'", '')
            words = words.replace('_', ' ')
            words = words.replace('&', ' and ')
            words = words.replace('  ', ' ')
            words = words.replace('- ', ' ')
            words = words.replace('-', ' ')
            new_l = fid+' '+words
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

    process_txt(train_path)
    process_txt(dev_path)
    process_txt(test_path)

if __name__ == '__main__':
    main()
