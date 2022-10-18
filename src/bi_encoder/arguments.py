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

    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    sentence_pooling_method: str = field(default='cls')
    normlized: bool = field(default=False)


@dataclass
class DataArguments:
    sample_neg_from_topk: int = field(
        default=200, metadata={"help": "sample negatives from top-k"}
    )
    teacher_score_files: str = field(
        default=None, metadata={"help": "Path to score_file for distillation"}
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
        default=None, metadata={"help": "Path to negative"}
    )
    test_query_id_file: Union[str] = field(
        default=None, metadata={"help": "Path to query data"}
    )

    prediction_save_path: Union[str] = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
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
class RetrieverTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=1.0)
    contrastive_loss_weight: Optional[float] = field(default=0.0)
