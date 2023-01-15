import argparse
import collections

from msmarco_eval import compute_metrics_from_files


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--score_file", type=str, default=None)
    parser.add_argument("--qrels_file", type=str, default=None)

    return parser.parse_args()


def score_to_msmarco(score_file, ranking_file):
    all_scores = collections.defaultdict(list)
    for line in open(score_file):
        if len(line.strip()) == 0:
            continue
        qid, did, score = line.strip().split()
        score = score.strip('[]')
        score = float(score)
        all_scores[qid].append([did, score])

    with open(ranking_file, 'w', encoding='utf-8') as f:
        for qid, candidates in all_scores.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            for rank, cand in enumerate(candidates):
                f.write(qid + '\t' + cand[0] + f'\t{rank + 1}\n')


if __name__ == "__main__":
    args = get_args()

    ranking_file = args.score_file + '.ranked_results'
    score_to_msmarco(args.score_file, ranking_file)
    metrics = compute_metrics_from_files(args.qrels_file, ranking_file)

    print('#####################')
    for x, y in (metrics):
        print('{}: {}'.format(x, y))
    print('#####################')
