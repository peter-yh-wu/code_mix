import os

def remove_lang_tags(line):
    l_list = line.split()
    new_l_list = []
    for word in l_list:
        new_word = word.split('_')[0]
        new_l_list.append(new_word)
    new_l = ' '.join(new_l_list)
    return new_l

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    raw_text_dir = os.path.join(data_dir, 'raw_text')
    new_text_dir = os.path.join(data_dir, 'text')
    if not os.path.exists(new_text_dir):
        os.makedirs(new_text_dir)

    raw_text_files = os.listdir(raw_text_dir)
    raw_text_files = [f for f in raw_text_files if f.endswith('.txt')]
    raw_text_paths = [os.path.join(raw_text_dir, f) for f in raw_text_files]

    for i, raw_text_path in enumerate(raw_text_paths):
        with open(raw_text_path, 'r') as inf:
            curr_lines = inf.readlines()
        new_lines = []
        for l in curr_lines:
            l_list = l.strip().split()
            new_l_list = []
            for word in l_list:
                if word[0] != '[':
                    new_l_list.append(word)
            new_l_list = new_l_list[3:]
            new_l = ' '.join(new_l_list)
            new_l = remove_lang_tags(new_l)
            new_lines.append(new_l)
        curr_raw_text_file = raw_text_files[i]
        fid = curr_raw_text_file.split('_')[0]
        new_text_file = fid+'_barebones.txt'
        new_text_path = os.path.join(new_text_dir, new_text_file)
        with open(new_text_path, 'w+') as ouf:
            for l in new_lines:
                ouf.write('%s\n' % l)

if __name__ == '__main__':
    main()