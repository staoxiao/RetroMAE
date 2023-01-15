import argparse

from bi_encoder.faiss_retriever import search_by_faiss
from msmarco_eval import compute_metrics_from_files


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_reps_path", type=str, default=None)
    parser.add_argument("--passage_reps_path", type=str, default=None)
    parser.add_argument("--qrels_file", type=str, default=None)
    parser.add_argument("--ranking_file", type=str, default=None)
    parser.add_argument("--use_gpu", action='store_true', default=False)
    parser.add_argument("--depth", type=int, default=1000)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    search_by_faiss(args.query_reps_path, args.passage_reps_path, args.ranking_file, batch_size=512, depth=1000,
                    use_gpu=args.use_gpu)

    if args.qrels_file is not None:
        metrics = compute_metrics_from_files(args.qrels_file, args.ranking_file)

        print('#####################')
        for x, y in (metrics):
            print('{}: {}'.format(x, y))
        print('#####################')
