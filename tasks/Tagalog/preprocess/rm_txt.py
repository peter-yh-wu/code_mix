'''Removes (()) symbol and empty lines'''

import os

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    txt_dir = os.path.join(data_dir, 'txt')
    scripted_txt_path = os.path.join(txt_dir, 'script.txt')

    with open(scripted_txt_path, 'r') as inf:
        script_lines = inf.readlines()

    new_lines = []
    for i, l in enumerate(script_lines):
        l = l.strip()
        l_list = l.split()
        new_l_list = []
        for w in l_list:
            if w != '(())':
                new_l_list.append(w)
        new_l = ' '.join(new_l_list)
        if len(new_l_list) > 1:
            new_lines.append(new_l)

    with open(scripted_lids_path, 'w') as ouf:
        for l in new_lines:
            ouf.write('%s\n' % l)

if __name__ == "__main__":
    main()