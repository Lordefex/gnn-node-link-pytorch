# Graph Neural Networks: Node Classification & Link Prediction (Demo)
![License](https://img.shields.io/github/license/KonNik88/gnn-node-link-pytorch)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-green)

Minimal but complete project on **Graph Neural Networks (GNNs)** with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). This is the **demo tier** (Cora/PubMed) to validate code, training loop, and metrics before moving to a real-world graph project (e.g., user–item recommendations with GNN).

## Idea
Show how GNNs work on **citation graphs** (Cora, PubMed) for two canonical tasks:
- **Node classification** — predict a paper’s research field using graph structure + features.
- **Link prediction** — predict whether a citation edge should exist between two papers.

We compare simple baselines (LogReg, MLP) to structural models (GCN, GraphSAGE, GAT) and include **explainability** (GNNExplainer). The goal is a compact, reproducible, portfolio‑ready demo.

---

## What's inside
- **Datasets**: Cora (default), PubMed
- **Tasks**:
  - Node classification
  - Link prediction
- **Models**:
  - Baselines: Logistic Regression / MLP
  - GCN (Graph Convolutional Network)
  - GraphSAGE
  - GAT (Graph Attention Network)
- **Interpretability**: GNNExplainer
- **Reproducibility**: fixed splits, random seed, environment.yml

---

## Project structure
```
.
├─ src/
│  ├─ data.py           # dataset loading
│  ├─ models/           # GCN, GraphSAGE, GAT
│  ├─ train_node.py     # node classification training
│  ├─ train_link.py     # link prediction training
│  ├─ explain.py        # GNNExplainer examples
│  └─ utils.py          # helpers (seed, logger, early stopping)
├─ configs/             # configs (Hydra or argparse)
├─ notebooks/           # EDA and visualization
├─ artifacts/           # checkpoints, logs
├─ environment.yml
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## Quickstart

1) Create environment
```bash
conda env create -f environment.yml
conda activate gnn
```

2) Install requirements
```bash
pip install -r requirements.txt
```

3) Run node classification on Cora (GCN)
```bash
python -m src.train_node --dataset Cora --model gcn --hid 64 --lr 0.003 --dropout 0.5 --epochs 300 --seed 42
```

4) Run link prediction on Cora
```bash
python -m src.train_link --dataset Cora --model gcn --hid 64 --epochs 200
```

> Tip: For **PubMed**, increase `--epochs` and consider lowering `--lr` a bit.

---

## Expected results (reference)

| Model     | Cora (Acc) | PubMed (Acc) |
|-----------|------------|--------------|
| MLP (bow) | ~0.58–0.62 | ~0.70–0.73   |
| GCN       | ~0.80–0.83 | ~0.78–0.81   |
| GraphSAGE | ~0.80–0.84 | ~0.79–0.82   |
| GAT       | ~0.82–0.85 | ~0.79–0.82   |

For link prediction on Cora, **ROC-AUC > 0.90** is typically achievable (dot-product scoring over node embeddings).

---

## Reproducibility
- `seed=42`
- Fixed Planetoid dataset splits
- Tested with:
  - Python 3.10
  - PyTorch 2.x
  - PyTorch Geometric 2.x

---

## Next steps (real-world tier)
- Graph recommenders on **Goodbooks-10k** (user–item bipartite graph), link prediction with GraphSAGE.
- PPI or molecular graphs (QM9) for property prediction.
- OGB datasets with neighbor sampling (scalability).
- Advanced explainability (PGExplainer) and CI (lint + tests).

---

## License
[MIT](LICENSE)
