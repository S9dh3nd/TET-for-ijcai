# Pytorch Graph Attention Network

This is a pytorch implementation of the Graph Attention Network (GAT)

To avoid repeated calculation, the GAT code is the original code, and its input is the result of ELCO-GCN output

## GAT-ELCO Run the demo

```bash
python train.py
```

## GAT-ELCO Data

You can copy *.cites, *.content, and *.extra from '..\ELCO\ELCO-GCN(run first)\gcn' to '..\ELCO\ELCO-GAT\gat\data\*'

You can specify a dataset as follows:

change 'def load_data(path="./data/citeseer/", dataset="citeseer"):' in utils.py into corresponding dataset (cora, citeseer, pubmed)
change 'idx_train = list(range(120)) + extra_train' in utils.py into corresponding dataset (cora: 140, citeseer: 120, pubmed: 60)

# Requirements

pyGAT relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).

