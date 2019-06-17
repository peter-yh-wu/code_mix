import os

def rm_bracket_words(l):
    l_list = l.split()
    new_l_list = []
    for w in l_list:
        if w[0] != '<':
            new_l_list.append(w)
    new_l = ' '.join(new_l_list)
    return new_l

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    script_dir = os.path.join(data_dir, 'scripted')
    script_txt_dir = os.path.join(script_dir, 'transcription')

    script_txt_files = os.listdir(script_txt_dir)
    script_txt_files = [f for f in script_txt_files if f.endswith('.txt')]
    script_txt_paths = [os.path.join(script_txt_dir, f) for f in script_txt_files]

    new_txt_dir = os.path.join(data_dir, 'txt')
    new_script_txt_path = os.path.join(new_txt_dir, 'script.txt')

    script_lines = []
    for i, text_path in enumerate(script_txt_paths):
        text_file = script_txt_files[i]
        fid = text_file[:-4]
        with open(text_path, 'r') as inf:
            curr_lines = inf.readlines()
        l = curr_lines[1].strip()
        l = rm_bracket_words(l)
        l = fid+' '+l
        script_lines.append(l)

    with open(new_script_txt_path, 'w+') as ouf:
        for l in script_lines:
            ouf.write('%s\n' % l)

if __name__ == '__main__':
    main()