import argparse
import csv
import os

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='submission.csv', help='csv file with transcripts')
    parser.add_argument('--save-directory', type=str, default='output/baseline/v1', help='output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    beam_path = os.path.join(args.save_directory, args.file)
    beams = load_beams(beam_path)
    print('%d beam groups, each of size %d' % (len(beams), len(beams[0])))

    rand_beams = []
    for beam in beams:
        rand_beams.append(beam[0])
    rand_path = os.path.join(args.save_directory, 'rand.csv')
    with open(rand_path, 'w', newline='') as f:
        w = csv.writer(f)
        for i, t in enumerate(rand_beams):
            w.writerow([i+1, t])


if __name__ == "__main__":
    main()