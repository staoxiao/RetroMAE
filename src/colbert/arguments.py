import os
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from typing import Optional, Union, List

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

    add_pq: bool = field(default=False)
    num_pq: int = field(default=0)
    num_codebook: int = field(default=0)
    num_codeword: int = field(default=256)

    init_faiss_file: str = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    def __post_init__(self):
        if self.projection_out_dim != 768:
            self.add_pooler = True
        if self.num_codebook != 0:
            self.add_pq = True


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

        print(self.teacher_score_files)
        if self.neg_file and not os.path.exists(self.neg_file):
            print('there is no neg_file. We will randomly sample negatives')
            self.neg_file = None


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=1.0)
    l2_weight: Optional[float] = field(default=0.002)
    pq_learning_rate: Optional[float] = field(default=1e-5)


def to_dict(args):
    """
    Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
    the token values by removing their value.
    """
    d = asdict(args)
    for k, v in d.items():
        if isinstance(v, Enum):
            d[k] = v.value
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
            d[k] = [x.value for x in v]
        if k.endswith("_token"):
            d[k] = f"<{k.upper()}>"
    return d

def save_args_to_json(args, output_file):
    args_dict = to_dict(args)
    with open(output_file, 'w') as f:
        f.write(json.dumps(args_dict))
