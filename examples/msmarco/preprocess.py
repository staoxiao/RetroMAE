import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--use_title", action='store_true', default=False)
    return parser.parse_args()


def save_to_json(input_file, output_file, id_file):
    with open(output_file, 'w', encoding='utf-8') as f, open(id_file, 'w', encoding='utf-8') as fid:
        cnt = 0
        for line in open(input_file, encoding='utf-8'):
            line = line.strip('\n').split('\t')
            if len(line) == 2:
                data = {"id": line[0], 'text': line[1]}
            else:
                data = {"id": line[0], 'title': line[1], 'text': line[2]}
            f.write(json.dumps(data) + '\n')
            fid.write(line[0] + '\t' + str(cnt) + '\n')
            cnt += 1


def preprocess_qrels(train_qrels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in open(train_qrels, encoding='utf-8'):
            line = line.strip().split('\t')
            f.write(line[0] + '\t' + line[2] + '\n')


def tokenize_function(examples):
    if 'title' in examples and args.use_title:
        content = []
        for title, text in zip(examples['title'], examples['text']):
            content.append(title + tokenizer.sep_token + text)
        return tokenizer(content, add_special_tokens=False, truncation=True, max_length=max_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)
    else:
        return tokenizer(examples["text"], add_special_tokens=False, truncation=True, max_length=max_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)


args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
max_length = args.max_seq_length

if __name__ == '__main__':
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'dev_query')).mkdir(parents=True, exist_ok=True)

    preprocess_qrels('./data/qrels.train.tsv', os.path.join(args.output_dir, 'train_qrels.txt'))

    save_to_json('./data/corpus.tsv', './data/corpus.json', os.path.join(args.output_dir, 'corpus/mapping_id.txt'))
    corpus = load_dataset('json', data_files='./data/corpus.json', split='train')
    corpus = corpus.map(tokenize_function, num_proc=8, remove_columns=["title", "text"], batched=True)
    corpus.save_to_disk(os.path.join(args.output_dir, 'corpus'))
    print('corpus dataset:', corpus)

    save_to_json('./data/train.query.txt', './data/train.query.json',
                 os.path.join(args.output_dir, 'train_query/mapping_id.txt'))
    train_query = load_dataset('json', data_files='./data/train.query.json', split='train')
    train_query = train_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    train_query.save_to_disk(os.path.join(args.output_dir, 'train_query'))
    print('train query dataset:', corpus)

    save_to_json('./data/dev.query.txt', './data/dev.query.json',
                 os.path.join(args.output_dir, 'dev_query/mapping_id.txt'))
    dev_query = load_dataset('json', data_files='./data/dev.query.json', split='train')
    dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query'))
    print('dev query dataset:', corpus)
