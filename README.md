# SVD-DeLaN Transfer Learning for Soft Robot Dynamics

This project trains a dynamics model for a soft robot using a transfer-learning setup based on **Deep Lagrangian Networks (DeLaN)**, with an optional **SVD-based parameterization** for compressed fine-tuning.

The code loads a pretrained DeLaN model, adapts it to a new dataset, and trains it to predict the next robot state from the current state and input.

## Overview

The implementation combines ideas from:

- *Physics-informed Neural Networks to Model and Control Robots: a Theoretical and Experimental Investigation : https://arxiv.org/abs/2305.05375*
- *SVD-PINNs: Transfer Learning of Physics-Informed Neural Networks via Singular Value Decomposition : https://arxiv.org/abs/2211.08760*

In this repository, the pretrained physics-informed dynamics model is reused and fine-tuned on new data. When SVD mode is enabled, dense pretrained weights are converted into an SVD form so that only selected parameters are updated during training.

The `utils.py` file and the base DeLaN implementation are obtained from : https://github.com/jingyueliu6/PINNs_LNN_HNN 


## Data Format

Each row in the dataset is structured as:

```text
[q, dq, input, q_next, dq_next]
```

## Files

### Training script
Main script for:
- loading and preprocessing data
- loading a pretrained model
- converting parameters to SVD form if enabled
- training and evaluating the model
- plotting losses
- saving results and trained weights

### `DeLaN_model_svd.py`
SVD-based version of the DeLaN model.  
Includes:
- SVD parameterization utilities
- selective parameter masking
- loading dense pretrained weights into SVD form
- SVD-based MLP layers
- forward and inverse dynamics models
- training loss functions

## Requirements

- Python 3.9+
- numpy
- jax
- haiku
- optax
- matplotlib


