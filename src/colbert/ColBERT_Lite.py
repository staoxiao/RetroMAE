import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput

from colbert.arguments import ModelArguments, \
    RetrieverTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DensePooler(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(DensePooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}
        self.output_dim = output_dim

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError



class Quantization(nn.Module):
    def __init__(self, num_pq: int = 1024, num_codebook: int = 128, num_codeword: int = 256, subvec_size: int = 6,
                 use_rotate: bool = False,
                 init_faiss_file: str = None):
        super(Quantization, self).__init__()

        self.pq_layers = []
        self.rotate = None
        self.num_pq = num_pq

        if init_faiss_file is not None and os.path.exists(init_faiss_file):
            logger.info(f'Loading pq from {init_faiss_file}')
            if os.path.exists(os.path.join(init_faiss_file, 'rotate.npy')):
                rotate = np.load(os.path.join(init_faiss_file, 'rotate.npy'))
                self.rotate = nn.Parameter(torch.FloatTensor(rotate), requires_grad=False)
                use_rotate = True
            if os.path.exists(os.path.join(init_faiss_file, 'ivf.npy')):
                ivf = np.load(os.path.join(init_faiss_file, 'ivf.npy'))
                self.pq_assgin_layer = nn.Parameter(torch.FloatTensor(ivf), requires_grad=True)

            codebooks = np.load(os.path.join(init_faiss_file, 'codebooks.npy'))
            num_codebook, num_codeword, subvec_size = np.shape(codebooks)
            self.num_codebook = num_codebook
            self.codebooks = nn.Parameter(torch.FloatTensor(codebooks), requires_grad=True)

        else:
            emb_size = int(num_codebook * subvec_size)
            self.num_codebook = num_codebook
            self.pq_assgin_layer = nn.Parameter(torch.empty(num_pq, emb_size)
                                                .uniform_(-0.1, 0.1)).type(torch.FloatTensor)
            if use_rotate:
                self.rotate = nn.Parameter(torch.empty(emb_size, emb_size)
                                           .uniform_(-0.1, 0.1)).type(torch.FloatTensor)

            self.codebooks = nn.Parameter(torch.empty(num_codebook, num_codeword, subvec_size)
                                          .uniform_(-0.1, 0.1)).type(torch.FloatTensor)

        self._config = {'num_pq': num_pq, 'num_codebook': num_codebook, 'num_codeword': num_codeword,
                        'subvec_size': subvec_size,
                        'use_rotate': use_rotate}

    def load(self, model_dir: str):
        pq_path = os.path.join(model_dir, 'pq.pt')
        if pq_path is not None:
            if os.path.exists(pq_path):
                logger.info(f'Loading pq from {pq_path}')
                state_dict = torch.load(pq_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training pq from scratch")
        return

    def save_pq(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pq.pt'))
        with open(os.path.join(save_path, 'pq_config.json'), 'w') as f:
            json.dump(self._config, f)

    def rotate_vec(self,
                   vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def pq_selection(self, vecs):
        # vecs: B D     pq_assgin: N D
        ip = torch.matmul(vecs, self.pq_assgin_layer.T)  # B N
        norm1 = torch.sum(vecs ** 2, dim=-1, keepdim=True)  # B 1
        norm2 = torch.sum(self.pq_assgin_layer ** 2, dim=-1, keepdim=True)  # N 1
        simi = norm1 + norm2.T - 2 * ip
        simi = -simi
        pq_index = simi.max(dim=-1)[1]

        return simi, self.pq_assgin_layer[pq_index]

    def code_selection(self, vecs):
        vecs = vecs.view(vecs.size(0), self.codebooks.size(0), -1)
        codebooks = self.codebooks.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        proba = - torch.sum((vecs.unsqueeze(-2) - codebooks) ** 2, -1)
        assign = F.softmax(proba / 0.01, -1)
        return assign

    def STEstimator(self, assign):
        index = assign.max(dim=-1, keepdim=True)[1]
        assign_hard = torch.zeros_like(assign, device=assign.device, dtype=assign.dtype).scatter_(-1, index, 1.0)
        return assign_hard.detach() - assign.detach() + assign

    def quantized_vecs(self, assign):
        codebooks = self.codebooks.unsqueeze(0).expand(assign.size(0), -1, -1, -1)
        quantized_vecs = torch.matmul(assign, codebooks).squeeze(2)
        quantized_vecs = quantized_vecs.view(assign.size(0), -1)
        return quantized_vecs

    def forward(self, vectors):
        vectors = self.rotate_vec(vectors)
        pq_simi, pq_centers = self.pq_selection(vectors)

        codeword_assgin = self.code_selection(vectors - pq_centers)
        assign = self.STEstimator(codeword_assgin)
        assign = assign.unsqueeze(2)
        quantized_vecs = self.quantized_vecs(assign)

        quantized_vecs = torch.nn.functional.normalize(pq_centers + quantized_vecs, dim=-1)

        return pq_simi, pq_centers, codeword_assgin, quantized_vecs


class ColBertModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 pq: nn.Module = None,
                 untie_encoder: bool = False,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 temperature: float = 1.0,
                 contrastive_loss_weight: float = 1.0,
                 l2_weight: float = 0.002
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.pq = pq
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl = nn.KLDivLoss(reduction="mean")
        self.untie_encoder = untie_encoder

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.contrastive_loss_weight = contrastive_loss_weight

        self.l2_weight = l2_weight

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_reps = psg_out.last_hidden_state

        if self.pooler is not None:
            p_reps = self.pooler(p=p_reps)
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_reps = qry_out.last_hidden_state

        if self.pooler is not None:
            q_reps = self.pooler(q=q_reps)
        if self.normlized:
            q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
        return q_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps, q_mask, p_mask):
        if len(p_reps.size()) == 3 and p_reps.size(0) == q_reps.size(0):
            # for inference
            # p_reps: B P D
            # q_reps: B Q D
            scores = torch.matmul(q_reps, p_reps.transpose(-2, -1)).unsqueeze(1)  # B 1 Q P
            p_mask = p_mask[:, None, None, :, ]  # B 1 1 P
        elif len(p_reps.size()) == 4:
            # p_reps: B N P D
            # q_reps: B Q D
            q_reps = q_reps.unsqueeze(1)  # B 1 Q D
            scores = torch.matmul(q_reps, p_reps.transpose(-2, -1))  # B N Q P
            p_mask = p_mask.view(q_reps.size(0), -1, p_mask.size(-1))  # B N P
            p_mask = p_mask.unsqueeze(2)  # B N 1 P
        elif len(p_reps.size()) == 3:
            # p_reps: BN P D
            # q_reps: B Q D
            q_reps = q_reps.unsqueeze(1)  # B 1 Q D
            p_reps = p_reps.unsqueeze(0)  # 1 BN P D
            scores = torch.matmul(q_reps, p_reps.transpose(-2, -1))  # B BN Q P
            p_mask = p_mask[None, :, None, :, ]  # 1 BN 1 P
        else:
            raise NotImplementedError

        q_mask = q_mask[:, None, :, None]  # B 1 Q 1
        scores = scores * q_mask

        temp_mask = (1 - p_mask) * -1000
        scores = scores + temp_mask

        values, _ = torch.max(scores, dim=-1)  # B N Q
        return torch.sum(values, dim=-1)  # B N

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler


    @staticmethod
    def load_pq(model_weights_file, **config):
        pq = Quantization(**config)
        pq.load(model_weights_file)
        return pq

    @staticmethod
    def build_pq(model_args):
        pq = Quantization(
            num_pq=model_args.num_pq,
            num_codebook=model_args.num_codebook,
            num_codeword=model_args.num_codeword,
            subvec_size=model_args.projection_out_dim // model_args.num_codebook,
            init_faiss_file=model_args.init_faiss_file
        )
        pq.load(model_args.model_name_or_path)
        return pq

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        # with torch.no_grad():
        q_reps, q_weight = self.encode_query(query)  # B Q D
        p_reps, p_weight = self.encode_passage(passage)  # BN P D

        if self.training:
            scores = self.compute_similarity(q_reps, p_reps, query['attention_mask'],
                                             passage['attention_mask']) / self.temperature
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)

            if self.pq:
                rotate_q_reps = self.pq.rotate_vec(q_reps)

                pq_simi, p_centers, codeword_assgin, quantized_p_reps = self.pq(p_reps.view(-1, p_reps.size(-1)))
                quantized_p_reps = quantized_p_reps.view(-1, p_reps.size(-2), p_reps.size(-1))
                quantized_scores = self.compute_similarity(rotate_q_reps, quantized_p_reps, query['attention_mask'],
                                                           passage['attention_mask']) / self.temperature

                l2_loss = torch.sum((self.pq.rotate_vec(p_reps) - quantized_p_reps) ** 2, dim=-1) * passage[
                    'attention_mask']
                l2_loss = torch.sum(l2_loss) / torch.sum(passage['attention_mask'])

                # constrastive
                # loss = loss + self.compute_loss(quantized_scores, target) + self.l2_weight * l2_loss

                # distill loss
                preds_smax = F.softmax(quantized_scores, dim=-1)
                true_smax = F.softmax(scores, dim=-1)
                preds_smax = preds_smax + 1e-6
                preds_log = torch.log(preds_smax)

                loss = loss + torch.mean(-torch.sum(true_smax * preds_log, dim=1))  + self.l2_weight * l2_loss


            return EncoderOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps, )

        else:
            if self.pq:
                q_reps = self.pq.rotate_vec(q_reps)
                _, _, _, quantized_p_reps = self.pq(p_reps.view(-1, p_reps.size(-1)))
                p_reps = quantized_p_reps.view(p_reps.size(0), p_reps.size(1), -1)
            scores = self.compute_similarity(q_reps, p_reps, query['attention_mask'],
                                             passage['attention_mask']).squeeze(1)

            return EncoderOutput(
                loss=None,
                scores=scores
            )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:

                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        if model_args.add_pq:
            pq = cls.build_pq(model_args)
        else:
            pq = None


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            pq=pq,
            untie_encoder=model_args.untie_encoder,
            normlized=model_args.normlized,
            sentence_pooling_method=model_args.sentence_pooling_method,
            temperature=train_args.temperature,
            l2_weight=train_args.l2_weight
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            normlized,
            sentence_pooling_method,
            init_faiss_file,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        pq_weights = os.path.join(model_name_or_path, 'pq.pt')
        pq_config = os.path.join(model_name_or_path, 'pq_config.json')
        if os.path.exists(pq_weights) and os.path.exists(pq_config):
            logger.info(f'found pq weight and configuration')
            with open(pq_config) as f:
                pq_config_dict = json.load(f)
            pq = cls.load_pq(model_name_or_path, **pq_config_dict)
        elif os.path.exists(init_faiss_file):
            pq = Quantization(init_faiss_file=init_faiss_file)
        else:
            pq = None


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            pq=pq,
            untie_encoder=untie_encoder,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
            if self.lm_p.config:
                self.lm_p.config.save_pretrained(output_dir)
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

        if self.pq:
            self.pq.save_pq(output_dir)