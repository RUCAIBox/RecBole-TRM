# Experiments

## Experimental Settings

**Dataset:** [MovieLens-1M](https://grouplens.org/datasets/movielens/)

**Filtering:** Remove interactions with a rating score of less than 3

**Evaluation:** ratio-based 8:1:1, full sort

**Metrics:** Recall, NGCG, MRR, HR, Precision

**TopK:** 10, 20

**Properties:**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
val_interval:
    rating: "[3,inf)"
unused_col: 
    inter: [rating]

# training and evaluation
epochs: 100
train_batch_size: 2048
valid_metric: MRR@10
eval_batch_size: 4096
```

For fairness, we restrict items' embedding dimension and the number of layers as following. Please adjust the name of the corresponding args of different models.
```
embedding_size: 64
n_layers: 2
```

## Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| ml-1m      | 6,040  | 3,629  | 836,478       | 96.18%   |


## Hyper-parameters

|              | Best hyper-parameters                                        | Tuning range                                                 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **TiSASRec**      | learning_rate=0.001| learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005] |
| **SSE-PT**      | learning_rate=0.002| learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005] |
| **LightSANs**      | learning_rate=0.002| learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005] |
| **gMLP**      | learning_rate=0.002| learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005] |
| **CORE**      | learning_rate=0.001| learning_rate choice [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005] |
