import logging

import torch
import torch.nn.functional as F
from pretrain.arguments import ModelArguments
from pretrain.enhancedDecoder import BertLayerForDecoder
from torch import nn
from transformers import BertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

logger = logging.getLogger(__name__)


class DupMAEForPretraining(nn.Module):
    def __init__(
            self,
            bert: BertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(DupMAEForPretraining, self).__init__()
        self.lm = bert

        self.decoder_embeddings = self.lm.bert.embeddings
        self.c_head = BertLayerForDecoder(bert.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def decoder_mlm_loss(self, sentence_embedding, decoder_input_ids, decoder_attention_mask, decoder_labels):
        sentence_embedding = sentence_embedding.view(decoder_input_ids.size(0), 1, -1)
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([sentence_embedding, decoder_embedding_output[:, 1:]], dim=1)

        query_embedding = sentence_embedding.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=query_embedding)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return loss

    def ot_embedding(self, logits, attention_mask):
        mask = (1 - attention_mask.unsqueeze(-1)) * -1000
        reps, _ = torch.max(logits + mask, dim=1)  # B V
        return reps

    def decoder_ot_loss(self, ot_embedding, bag_word_weight):
        input = F.log_softmax(ot_embedding, dim=-1)
        bow_loss = torch.mean(-torch.sum(bag_word_weight * input, dim=1))
        return bow_loss

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels,
                bag_word_weight):
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, 0]
        mlm_loss = self.decoder_mlm_loss(cls_hiddens, decoder_input_ids, decoder_attention_mask, decoder_labels)

        ot_embedding = self.sparse_embedding(lm_out.logits[:, 1:], encoder_attention_mask[:, 1:])
        bow_loss = self.decoder_ot_loss(ot_embedding, bag_word_weight=bag_word_weight)

        loss = mlm_loss + self.model_args.bow_loss_weight * bow_loss + lm_out.loss

        return (loss, )



    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model
