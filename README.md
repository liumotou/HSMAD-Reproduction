## Overview

HSMAD: Heterophily-Driven Spectral and Manifold Learning for Graph Anomaly Detection

## Usage

### Command-line arguments

You can run the model by executing the following command:

```bash
python main.py --dataset weibo
```

### Arguments

- `--dataset` (str): Specify the dataset to use. Available options include `yelp`, `weibo`, `amazon`, `elliptic`, and `tolokers`.
- `--run` (int): The number of runs for the experiment. Default is 1.
- `--epoch` (int): Number of epochs to train the model. Default is 1000.
- `--patience` (int): Patience for early stopping. The model will stop training if there is no improvement for this many epochs. Default is 100.
- `--order` (int): The order of filters. Default is 2.
- `--q` (float): The quantile parameter. Default is 0.5.

### Example

To run the model with the `weibo` dataset, 10 runs, 1000 epochs, patience of 100, execute:

```bash
python main.py --dataset weibo --run 10 --epoch 1000 --patience 100 --order 2 --q 0.5
```

This will train the model using the specified hyperparameters and dataset.

### Weibo ablation study

The three commands below reproduce the complete-model and component-removal
settings used for the Weibo columns of Table 2 and Table 5. Each command runs
ten random initializations; `full` is the original HSMAD model.

```bash
# HSMAD
python main.py --dataset weibo --gpu 0 --run 10 --epoch 1000 --patience 100 --hid_dim 64 --order 2 --q 0.5 --ablation full

# w/o HWSF: retain only the HRMU manifold branch
python main.py --dataset weibo --gpu 0 --run 10 --epoch 1000 --patience 100 --hid_dim 64 --order 2 --q 0.5 --ablation no_hwsf

# w/o HRMU: retain only the HWSF spectral branch
python main.py --dataset weibo --gpu 0 --run 10 --epoch 1000 --patience 100 --hid_dim 64 --order 2 --q 0.5 --ablation no_hrmu
```

### Output

The results of the experiment will include:

- Performance metrics (e.g., Recall, Pecision, F1-Macro, AUROC, AUPRC, G-Mean)
