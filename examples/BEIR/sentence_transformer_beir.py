import logging
import os
from typing import Optional, OrderedDict, Iterable, Union, Tuple

import torch
import torch.nn as nn
from beir.retrieval.models import SentenceBERT
from sentence_transformers import SentenceTransformer, __MODEL_HUB_ORGANIZATION__, __version__
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.util import snapshot_download

logger = logging.getLogger(__name__)

class SentenceTransformerForBEIR(SentenceTransformer):
    def __init__(self, model_name_or_path: Optional[str] = None, pooling_strategy: Optional[str] = None, modules: Optional[Iterable[nn.Module]] = None,
                 device: Optional[str] = None, cache_folder: Optional[str] = None):
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if cache_folder is None:
            cache_folder = os.getenv('SENTENCE_TRANSFORMERS_HOME')
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

                cache_folder = os.path.join(torch_cache_home, 'sentence_transformers')

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            # Old models that don't belong to any organization
            basic_transformer_models = ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2',
                                        'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1',
                                        'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased',
                                        'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased',
                                        'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased',
                                        'bert-base-multilingual-uncased', 'bert-base-uncased',
                                        'bert-large-cased-whole-word-masking-finetuned-squad',
                                        'bert-large-cased-whole-word-masking', 'bert-large-cased',
                                        'bert-large-uncased-whole-word-masking-finetuned-squad',
                                        'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base',
                                        'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased',
                                        'distilbert-base-german-cased', 'distilbert-base-multilingual-cased',
                                        'distilbert-base-uncased-distilled-squad',
                                        'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased',
                                        'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl',
                                        'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base',
                                        'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large',
                                        't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103',
                                        'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280',
                                        'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024',
                                        'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024',
                                        'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch',
                                        'xlm-roberta-large-finetuned-conll02-spanish',
                                        'xlm-roberta-large-finetuned-conll03-english',
                                        'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large',
                                        'xlnet-base-cased', 'xlnet-large-cased']

            if os.path.exists(model_name_or_path):
                # Load from path
                model_path = model_name_or_path
            else:
                # Not a path, load from hub
                if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if '/' not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path

                model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))

                # Download from hub with caching
                snapshot_download(model_name_or_path,
                                  cache_dir=cache_folder,
                                  library_name='sentence-transformers',
                                  library_version=__version__,
                                  ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])

            if os.path.exists(os.path.join(model_path, 'modules.json')):  # Load as SentenceTransformer model
                modules = self._load_sbert_model(model_path)
            else:  # Load with AutoModel
                modules = self._load_auto_model(model_path,pooling_strategy)

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules=modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)


    def _load_auto_model(self, model_name_or_path, pooling_strategy):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logging.warning(
            "No sentence-transformers model found with name {}. Creating a new one with {} pooling.".format(
                model_name_or_path, pooling_strategy))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling_strategy)
        return [transformer_model, pooling_model]

class SentenceBERTForBEIR(SentenceBERT):
    def __init__(self, model_path: Union[str, Tuple] = None, pooling_strategy: Optional[str] = None, sep: str = " ", **kwargs):
        self.sep = sep

        if isinstance(model_path, str):
            self.q_model = SentenceTransformerForBEIR(model_path, pooling_strategy)
            self.doc_model = self.q_model

        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformerForBEIR(model_path[0], pooling_strategy)
            self.doc_model = SentenceTransformerForBEIR(model_path[1], pooling_strategy)