import logging
import os
from pathlib import Path

import numpy as np
from bi_encoder import BiEncoderModel
from bi_encoder import BiTrainer
from bi_encoder.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from bi_encoder.data import TrainDatasetForBiE, PredictionDataset, BiCollator, PredictionCollator
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    if training_args.do_train:
        model = BiEncoderModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        model = BiEncoderModel.load(
            model_args.model_name_or_path,
            normlized=model_args.normlized,
            sentence_pooling_method=model_args.sentence_pooling_method
        )

    # Get datasets
    if training_args.do_train:
        train_dataset = TrainDatasetForBiE(args=data_args, tokenizer=tokenizer)
    else:
        train_dataset = None

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=BiCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        logging.info("*** Prediction ***")
        # if os.path.exists(data_args.prediction_save_path):
        #     raise FileExistsError(f"Existing: {data_args.prediction_save_path}. Please save to other paths")

        if data_args.corpus_file is not None:
            logging.info("*** Corpus Prediction ***")
            passage_path = os.path.join(data_args.prediction_save_path, 'passage_reps')
            Path(passage_path).mkdir(parents=True, exist_ok=True)

            trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=False)
            test_dataset = PredictionDataset(
                data_path=data_args.corpus_file, tokenizer=tokenizer,
                max_len=data_args.passage_max_len,
            )
            pred_scores = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(pred_scores)
                np.save(os.path.join(passage_path, 'passage.npy'), pred_scores)
                with open(os.path.join(passage_path, 'offset2passageid.txt'), "w") as writer:
                    for line in open(data_args.corpus_id_file):
                        cid, offset = line.strip().split('\t')
                        writer.write(f'{offset}\t{cid}\t\n')

        if data_args.test_query_file is not None:
            logging.info("*** Query Prediction ***")
            query_path = os.path.join(data_args.prediction_save_path, 'query_reps')
            Path(query_path).mkdir(parents=True, exist_ok=True)

            trainer.data_collator = PredictionCollator(tokenizer=tokenizer, is_query=True)
            test_dataset = PredictionDataset(
                data_path=data_args.test_query_file, tokenizer=tokenizer,
                max_len=data_args.query_max_len,
            )
            pred_scores = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset) == len(pred_scores)
                np.save(os.path.join(query_path, 'query.npy'), pred_scores)
                with open(os.path.join(query_path, 'offset2queryid.txt'), "w") as writer:
                    for line in open(data_args.test_query_id_file):
                        cid, offset = line.strip().split('\t')
                        writer.write(f'{offset}\t{cid}\t\n')


if __name__ == "__main__":
    main()
