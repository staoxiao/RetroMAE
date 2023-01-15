import os
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    sample_neg_from_topk: int = field(
        default=200, metadata={"help": "max k"}
    )
    prediction_topk: int = field(
        default=200, metadata={"help": "max k"}
    )

    corpus_file: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    corpus_id_file: Union[str] = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    train_query_file: Union[str] = field(
        default=None, metadata={"help": "Path to query data"}
    )
    train_query_id_file: Union[str] = field(
        default=None, metadata={"help": "Path to query data"}
    )
    train_qrels: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    neg_file: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    test_query_file: Union[str] = field(
        default=None, metadata={"help": "Path to query data"}
    )
    test_query_id_file: Union[str] = field(
        default=None, metadata={"help": "Path to query data"}
    )
    test_file: Union[str] = field(
        default=None, metadata={"help": "Path to test data"}
    )

    prediction_save_path: Union[str] = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    train_group_size: int = field(default=8)

    max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.corpus_file and not self.corpus_id_file:
            if not os.path.exists(os.path.join(self.corpus_file, 'mapping_id.txt')):
                raise FileNotFoundError(f'There is no mapping_id.txt in {self.corpus_file}')
            self.corpus_id_file = os.path.join(self.corpus_file, 'mapping_id.txt')

        if self.train_query_file and not self.train_query_id_file:
            if not os.path.exists(os.path.join(self.train_query_file, 'mapping_id.txt')):
                raise FileNotFoundError(f'There is no mapping_id.txt in {self.train_query_file}')
            self.train_query_id_file = os.path.join(self.train_query_file, 'mapping_id.txt')

        if self.test_query_file and not self.test_query_id_file:
            if not os.path.exists(os.path.join(self.test_query_file, 'mapping_id.txt')):
                raise FileNotFoundError(f'There is no mapping_id.txt in {self.test_query_file}')
            self.test_query_id_file = os.path.join(self.test_query_file, 'mapping_id.txt')


@dataclass
class CETrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=None)
