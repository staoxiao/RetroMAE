# RetroMAE
Implementation for [RetroMAE: Pre-training Retrieval-oriented Transformers via Masked Auto-Encoder
](https://arxiv.org/abs/2205.12035). 

## Released Models
We have uploaded some checkpoints to Huggingface Hub. 

| Model | Description | Link  |
|---|---|---|
|RetroMAE | Pre-trianed on the wikipedia and bookcorpus | [Shitao/RetroMAE](https://huggingface.co/Shitao/RetroMAE) | 
|RetroMAE_MSMARCO | Pre-trianed on the MSMARCO passage | [Shitao/RetroMAE_MSMARCO](https://huggingface.co/Shitao/RetroMAE_MSMARCO) | 
|RetroMAE_MSMARCO_finetune |Finetune the RetroMAE_MSMARCO on the MSMARCO passage data | [Shitao/RetroMAE_MSMARCO_finetune](https://huggingface.co/Shitao/RetroMAE_MSMARCO_finetune) | 
|RetroMAE_MSMARCO_distill | Finetune the RetroMAE on the MSMARCO passage data by minimizing the KL-divergence with the cross-encoder　| coming soon | 
|RetroMAE_BEIR | Finetune the RetroMAE on the MSMARCO passage data for BEIR (use the official negatives provided by BEIR)　| [Shitao/RetroMAE_BEIR](https://huggingface.co/Shitao/RetroMAE_BEIR) | 

You can load them easily using the identifier strings. For example:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('Shitao/RetroMAE')
```

## State of the Art Performance
RetroMAE can provide a strong initialization of dense retriever; after fine-tuned with in-domain data, it
gives rise to a high-quality supervised retrieval performance in the corresponding scenario. 
Besides, It substantially improves the pre-trained model's transferability, which helps to result in superior zero-shot performances on out-of-domain datasets.

### MSMARCO Passage
- Model pre-trained on wikipedia and bookcorpus:

| Model | MRR@10 | Recall@1000 |
|---|---|---|
|Bert | 0.346 | 0.964 |
|RetroMAE | **0.382** | **0.981** |

- Model pre-trained on MSMARCO:

| Model             | MRR@10 | Recall@1000 |
|-------------------|---|---|
| coCondenser         | 0.382 | 0.984 | 
| RetroMAE          | 0.393 | 0.985 | 
| RetroMAE(distillation) | **0.416** | **0.988** | 

### BEIR Benchemark

| Model             | Avg NDCG@10 (18 datasets) |
|-------------------|---|
| Bert         | 0.371 | 
| Condenser       | 0.407 | 
| RetroMAE       | **0.452** | 

## Installation
```
git clone https://github.com/staoxiao/RetroMAE.git
cd RetroMAE
pip install .
```
For development, install as editable:

```
pip install -e .
```

## Workflow
This repo includes two functions: pre-train and finetune. Firstly, train the RetroMAE on general dataset
 (or downstream dataset) with mask language modeling loss. Then finetune the RetroMAE on 
 downstream dataset with contrastive loss. To achieve a better performance, you also can finetune the 
 RetroMAE by distillation the scores provided by cross-encoder. **Detailed workflow please refer to our examples.** 

### Pretrain
```
python -m torch.distributed.launch --nproc_per_node 8 \
  -m pretrain.run \
  --output_dir {path to save ckpt} \
  --data_dir {your data} \
  --do_train True \
  --model_name_or_path bert-base-uncased 
```

### Finetune
```
python -m torch.distributed.launch --nproc_per_node 8 \
-m bi_encoder.run \
--output_dir {path to save ckpt} \
--model_name_or_path Shitao/RetroMAE \
--do_train  \
--corpus_file ./data/BertTokenizer_data/corpus \
--train_query_file ./data/BertTokenizer_data/train_query \
--train_qrels ./data/BertTokenizer_data/train_qrels.txt \
--neg_file ./data/train_negs.tsv 
```


## Examples

[Pre-train](examples/pretrain/README.md)  
[Finetune on MSMARCO Passage](examples/msmarco/README.md)  
[BEIR Benchemark](examples/BEIR/README.md)





