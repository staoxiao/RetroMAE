import argparse
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0)

    return parser.parse_args()


def create_book_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def book_tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)

    def book_pad_each_line(examples):
        blocks = []
        curr_block = []

        curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
        for sent in examples['input_ids']:
            if len(curr_block) >= curr_tgt_len:
                blocks.append(curr_block)
                curr_block = []
                curr_tgt_len = target_length if random.random() > short_seq_prob \
                    else random.randint(3, target_length)
            curr_block.extend(sent)
        if len(curr_block) > 0:
            blocks.append(curr_block)
        return {'token_ids': blocks}

    bookcorpus = load_dataset('bookcorpus', split='train')
    tokenized_bookcorpus = bookcorpus.map(book_tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=8, batched=True,
                                                    remove_columns=["input_ids"])
    return processed_bookcorpus


def create_wiki_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float = 0.0):
    import nltk
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def wiki_tokenize_function(examples):
        sentences = []
        for sents in examples['sentences']:
            sentences.append(
                tokenizer(sents, add_special_tokens=False, truncation=False, return_attention_mask=False,
                          return_token_type_ids=False)['input_ids'])
        return {"input_ids": sentences}

    def sentence_wiki(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        return {"sentences": sentences}

    def wiki_pad_each_line(examples):
        blocks = []
        for sents in examples['input_ids']:
            curr_block = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blocks.append(curr_block)
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
        return {'token_ids': blocks}

    wiki = load_dataset("wikipedia", "20200501.en", split="train")
    wiki = wiki.map(sentence_wiki, num_proc=8, remove_columns=["title", "text"])
    tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=8, batched=True, remove_columns=["sentences"])
    processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=8, batched=True, remove_columns=["input_ids"])

    return processed_wiki


def create_passage_data(tokenizer_name: str,
                        max_seq_length: int,
                        short_seq_prob: float = 0.0):
    import nltk
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def passage_tokenize_function(examples):
        if len(examples['title']) > 2:
            text = examples['title'] + ' ' + examples['text']
        else:
            text = examples['text']
        sentences = nltk.sent_tokenize(text)
        return tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                         return_token_type_ids=False)

    def passage_pad_each_line(examples):
        blocks = []
        for sents in examples['input_ids']:
            curr_block = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3,
                                                                                                 target_length)
            for sent in sents:
                if len(curr_block) >= curr_tgt_len:
                    blocks.append(curr_block)
                    curr_block = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
        return {'token_ids': blocks}

    msmarco = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
    msmarco = msmarco.remove_columns("docid")
    tokenized_msmarco = msmarco.map(passage_tokenize_function, num_proc=8, remove_columns=["text", "title"])
    processed_msmarco = tokenized_msmarco.map(passage_pad_each_line, num_proc=8, batched=True, batch_size=None,
                                              remove_columns=["input_ids"])

    return processed_msmarco


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.data == 'bert_data':
        print('download and preprocess wiki and bookcorpus:')
        wiki = create_wiki_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        book = create_book_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        dataset = concatenate_datasets([book, wiki])
        dataset.save_to_disk(args.output_dir)
    elif args.data == 'msmarco_passage':
        print('download and preprocess msmarco-passage:')
        dataset = create_passage_data(args.tokenizer_name, args.max_seq_length, args.short_seq_prob)
        dataset.save_to_disk(args.output_dir)
    else:
        raise NotImplementedError
