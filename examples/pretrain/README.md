# RetroMAE Pre-training

## Pre-train on general data

We use the same data as bert. You can download our checkpoint from huggingface hub: `Shitao/RetroMAE`.

### Prepare Data

```
python preprocess.py --data bert_data --tokenizer_name bert-base-uncased --output_dir pretrain_data/bert_data
```
This script will download and preprocess the dataset (wikipedia and bookcorpus), and then save them to `output_dir`.

### Pre-train

```
python -m torch.distributed.launch --nproc_per_node {number of gpus} \
  -m  pretrain.run \
  --output_dir {path to save model} \
  --data_dir {preprocessed data, e.g., pretrain_data/bert_data} \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 32 \
  --model_name_or_path bert-base-uncased \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 8 \
  --overwrite_output_dir True \
  --dataloader_num_workers 6 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5
```

## Pre-train on downstream data

Pre-train on the downstream data can achieve a better performance for the downstream task.
We take the msmarco passage as an example. You can download our checkpoint from huggingface hub: `Shitao/RetroMAE_MSMARCO`.

### Prepare Data

```bash
python preprocess.py --data msmarco_passage --tokenizer_name bert-base-uncased --output_dir pretrain_data/msmarco_passage
```

### Pre-train

```
python -m torch.distributed.launch --nproc_per_node {number of gpus} \
  -m  pretrain.run \
  --output_dir {path to save model} \
  --data_dir {preprocessed data, e.g., pretrain_data/msmarco_passage} \
  --do_train True \
  --save_steps 20000 \
  --per_device_train_batch_size 128 \
  --max_seq_length 150 \
  --model_name_or_path Shitao/RetroMAE \
  --fp16 True \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 20 \
  --overwrite_output_dir True \
  --dataloader_num_workers 6 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5
```





