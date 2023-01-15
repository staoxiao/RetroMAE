import argparse
import gzip
import json
import os

from beir import util
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    fout = open(args.output_file, 'w')

    ce_score_margin = 3  # Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = 5  # We used different systems to mine hard negatives. Number of hard negatives to add from each system

    triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
    msmarco_triplets_filepath = "msmarco-hard-negatives.jsonl.gz"
    if not os.path.isfile(msmarco_triplets_filepath):
        util.download_url(triplets_url, msmarco_triplets_filepath)

    with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
        for line in tqdm(fIn, total=502939):
            data = json.loads(line)

            # Get the positive passage ids
            pos_pids = [item['pid'] for item in data['pos']]
            pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin

            # Get the hard negatives
            neg_pids = set()
            for system_negs in data['neg'].values():
                negs_added = 0
                for item in system_negs:
                    if item['ce-score'] > ce_score_threshold:
                        continue

                    pid = item['pid']
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if len(pos_pids) > 0 and len(neg_pids) > 0:
                neg_pids = [str(x) for x in neg_pids]
                fout.write(str(data['qid']) + '\t' + ','.join(neg_pids) + '\n')
