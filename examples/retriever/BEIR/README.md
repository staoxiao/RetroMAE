# RetroMAE For BEIR


### Prepare Data

Download the MSMARCO data:
```
bash get_data.sh
python preprocess.py  --tokenizer_name bert-base-uncased --max_seq_length 150 --output_dir ./data/BertTokenizer_data
```

You can convert the official negatives to our format by following command:
```
pip install beir
python download_neg.py --output_file beir_neg.txt
```

### Train

We provide our checkpoint in huggingface hub: `Shitao/RetroMAE_BEIR`
You can train your model as following:

```
torchrun --nproc_per_node 8 \
-m bi_encoder.run \
--output_dir retromae_beir \
--model_name_or_path Shitao/RetroMAE \
--do_train  \
--corpus_file ./data/BertTokenizer_data/corpus \
--train_query_file ./data/BertTokenizer_data/train_query \
--train_qrels ./data/BertTokenizer_data/train_qrels.txt \
--neg_file beir_neg.txt \
--query_max_len 32 \
--passage_max_len 144 \
--fp16  \
--per_device_train_batch_size 128 \
--train_group_size 2 \
--sample_neg_from_topk 200 \
--learning_rate 1e-5 \
--num_train_epochs 10 \
--negatives_x_device  \
--dataloader_num_workers 6 
```

### Test
The test script is provided by https://github.com/SamuelYang1/SentMAE.
```
python beir_test.py \
--dataset nfcorpus \
--split test \
--batch_size 128 \
--model_name_or_path Shitao/RetroMAE_BEIR \
--pooling_strategy cls \
--score_function dot
```


