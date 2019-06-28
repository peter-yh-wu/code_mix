import os

def rm_lines(phase):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_dir = os.path.join(parent_dir, 'split')
    phase_file = '%s.txt' % phase
    phase_path = os.path.join(split_dir, phase_file)
    lid_file = '%s_lids.txt' % phase
    lid_path = os.path.join(split_dir, lid_file)
    with open(phase_path, 'r') as inf:
        y_lines = inf.readlines()
    with open(lid_path, 'r') as inf:
        lid_lines = inf.readlines()
    new_y_lines = []
    new_lid_lines = []
    for y, lid in zip(y_lines, lid_lines):
        y = y.strip()
        lid = lid.strip()
        if len(y.split()) > 1:
            new_y_lines.append(y)
            new_lid_lines.append(lid)
    new_ys_path = phase_path
    new_lid_path = lid_path
    with open(new_ys_path, 'w+') as ouf:
        for l in new_y_lines:
            ouf.write('%s\n' % l)
    with open(new_lid_path, 'w+') as ouf:
        for l in new_lid_lines:
            ouf.write('%s\n' % l)

def main():
    phases = ['train', 'test', 'dev']
    for phase in phases:
        rm_lines(phase)

if __name__ == '__main__':
    main()