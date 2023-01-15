import argparse
import collections


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ranking_file", type=str, default=None)
    parser.add_argument("--qrels_file", type=str, default=None)
    parser.add_argument("--output_neg_file", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    neg_data = collections.defaultdict(list)
    pos_data = collections.defaultdict(set)
    for line in open(args.qrels_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1]
        pos_data[qid].add(pos)
    for line in open(args.ranking_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1]
        if neg not in pos_data[qid]:
            neg_data[qid].append(neg)

    with open(args.output_neg_file, 'w') as f:
        for qid, negs in neg_data.items():
            line = f'{qid}\t{",".join(negs)}\n'
            f.write(line)
