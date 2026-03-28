# Diffusion Classifier
This is a minimal, cleaned-up version of HCS-DFC - Diffusion Classifier for HCS repository [link](https://github.com/gmum/HCS-DFC/tree/develop)
Implementation contains example workflow for 2 digit MNIST dataset.


## Environment Setup
We utilize **UV package manager** to manage dependencies, install UV by following instructions at official website https://github.com/astral-sh/uv   

To create uv venv and install all dependencies run:
```shell
uv sync
```

## Data Genereation
Start with generating **2-digit MNIST dataset** by running
- Normal 2-digit MNIST
```shell
uv run _generate_mnist/generate_2_digit_mnist.py
```
- 2-digit MNIST + occlusion (image might not contain all of the corresponding labels)
```shell
uv run _generate_mnist/generate_occluded_2_digit_mnist.py
```

## Training / Testing
This project utilizes [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to manage training, to fully utilize lightning additional CLI is required [WIP]


CNN Training
```shell
python lightning_training/train_cnn.py 
```
Diffusion Training
```shell
python lightning_training/train_diffusion.py 
```

## Code changes
put the necessary code changes in


Architectures -> `/models`

Lightning Wrappers containing looops etc. -> `/lightning_models`
