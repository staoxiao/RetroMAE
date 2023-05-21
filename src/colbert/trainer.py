
from transformers.trainer import *

import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer import Trainer, nested_detach
from transformers.trainer_utils import PredictionOutput, EvalPrediction
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ColBertTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None
        return (loss, logits, labels)

    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> PredictionOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label})


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and 'pq.' not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and 'pq.' not in n],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if 'pq.' in n],
                    "weight_decay": 0.0,
                    "lr": self.args.pq_learning_rate
                },

            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
