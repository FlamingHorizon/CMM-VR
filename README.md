# Cascaded Mutual Modulation for Visual Reasoning

Source code for Yiqun Yao; Jiaming Xu and Bo Xu. Cascaded Mutual Modulation for Visual Reasoning. 2018. EMNLP. [arxiv](https://arxiv.org/abs/1809.01943)

## Code Outline

This code is a fork from the code for "FiLM: Visual Reasoning with a General Conditioning Layer" available [here](https://github.com/ethanjperez/film).

We implement a new model: CMM on the basis of PG+EE and FiLM. Different from the original code, our model runs on multiple gpus.


## Setup and Training

Setup instructions for the CMM model are nearly the same as for PG+EE and FiLM.

First, follow the virtual environment setup [instructions](https://github.com/facebookresearch/clevr-iep#setup).

Second, follow the CLEVR data preprocessing [instructions](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr).

The below script has the hyper-parameters and settings to reproduce CMM CLEVR results:
```bash
sh scripts/train/cmm.sh
```

## Running models

The following scripts run trained models on CLEVR:
```bash
python scripts/run_model.py 
```
