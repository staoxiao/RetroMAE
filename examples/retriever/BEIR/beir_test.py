# Modifyed from BEIR Quick Example (https://github.com/beir-cellar/beir)

import argparse

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformer_beir import SentenceBERTForBEIR

import logging
import pathlib, os


def main(args):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

    #### Load the SBERT model and retrieve using dot-similarity
    model = DRES(SentenceBERTForBEIR(args.model_name_or_path, args.pooling_strategy), batch_size=args.batch_size)
    retriever = EvaluateRetrieval(model, score_function=args.score_function) # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--split", type=str, help="Tested on dev or test set")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--pooling_strategy", type=str, default='cls', help="'mean' or 'cls'")
    parser.add_argument("--score_function", type=str, default="dot",
        help="Metric used to compute similarity between two embeddings")
    args, _ = parser.parse_known_args()
    main(args)
