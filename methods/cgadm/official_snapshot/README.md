# Anomaly Detection through Conditional Diffusion Probability Modeling on Graphs

This repository is the official implementation of CGADM.

## Requirements

To install requirements:

```setup
conda env create -f gad.yaml
```

## Data Download

Download all the datasets from the following links and put them in the "dataset" folder:
[Download datasets](https://drive.google.com/drive/folders/1t1oWIDvCRp323FWbjwoVYOJM5sKBSfqP?usp=sharing)


## Training and Evaluation

To train the model(s) in the paper:

```setup
python main.py
```
> Output: the file "best_XXXXXX.tar" in the "data" folder

