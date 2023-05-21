import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from colbert.ColBERT_Lite import ColBertModel
from colbert.trainer import ColBertTrainer
from colbert.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments, save_args_to_json
from colbert.data import TrainDatasetForBiE, PredictionDataset, BiCollator

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
        model = ColBertModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = ColBertModel.load(
            model_args.model_name_or_path,
            normlized=model_args.normlized,
            sentence_pooling_method=model_args.sentence_pooling_method,
            init_faiss_file=model_args.init_faiss_file
        )

    # Get datasets
    if training_args.do_train:
        train_dataset = TrainDatasetForBiE(args=data_args, tokenizer=tokenizer)
    else:
        train_dataset = None

    trainer = ColBertTrainer(
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
            save_args_to_json(training_args, os.path.join(training_args.output_dir, 'train_args.json'))
            save_args_to_json(model_args, os.path.join(training_args.output_dir, 'model_args.json'))
            save_args_to_json(data_args, os.path.join(training_args.output_dir, 'data_args.json'))


    if training_args.do_predict:
        logging.info("*** Prediction ***")
        # if os.path.exists(data_args.prediction_save_path):
        #     raise FileExistsError(f"Existing: {data_args.prediction_save_path}. Please save to other paths")

        test_dataset = PredictionDataset(args=data_args, tokenizer=tokenizer)
        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        if trainer.is_world_process_zero():
            assert len(test_dataset) == len(pred_scores)
            Path(data_args.prediction_save_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(data_args.prediction_save_path, 'scores.txt'), "w") as writer:
                for pair, score in zip(test_dataset.test_data, pred_scores):
                    writer.write(f'{pair[0]}\t{pair[1]}\t{score}\n')



if __name__ == "__main__":
    main()
