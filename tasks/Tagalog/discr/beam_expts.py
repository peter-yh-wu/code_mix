import argparse
import csv
import os
import sys

from nltk.metrics import edit_distance

from discr_utils import *

def load_beams(path):
    raw_preds = []
    with open(path, 'r') as csvfile:
        raw_csv = csv.reader(csvfile)
        i = 0
        curr_preds = []
        for _, row in enumerate(raw_csv): # row is size-2 list
            curr_i = int(row[0])-1
            y_pred = row[1].strip() # string
            if i == curr_i:
                curr_preds.append(y_pred)
            else:
                raw_preds.append(curr_preds)
                curr_preds = [y_pred]
                i += 1
        raw_preds.append(curr_preds)
    return raw_preds


def get_best(beams, y_true):
    best_beams = []
    for b_group, y in zip(beams, y_true):
        best_i = 0
        best_cer = sys.maxsize
        for i, curr_beam in enumerate(b_group):
            curr_cer = edit_distance(curr_beam, y)/100
            if curr_cer < best_cer:
                best_i = i
                best_cer = curr_cer
        best_beams.append(b_group[best_i])
    return best_beams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='submission.csv', help='csv file with transcripts')
    parser.add_argument('--save-directory', type=str, default='output/baseline/v1', help='output directory')
    parser.add_argument('--mode', type=str, default='best', help='best or rand')
    return parser.parse_args()


def main():
    args = parse_args()
    beam_path = os.path.join(args.save_directory, args.file)
    beams = load_beams(beam_path)
    print('%d beam groups, each of size %d' % (len(beams), len(beams[0])))

    if args.mode == 'rand':
        rand_beams = []
        for beam in beams:
            rand_beams.append(beam[0])
        rand_path = os.path.join(args.save_directory, 'rand.csv')
        with open(rand_path, 'w', newline='') as f:
            w = csv.writer(f)
            for i, t in enumerate(rand_beams):
                w.writerow([i+1, t])
    else:
        _, y_test = load_fid_and_y_data('test')
        best_beams = get_best(beams, y_test)
        beams_path = os.path.join(args.save_directory, 'best.csv')
        with open(beams_path, 'w', newline='') as f:
            w = csv.writer(f)
            for i, t in enumerate(best_beams):
                w.writerow([i+1, t])


if __name__ == "__main__":
    main()