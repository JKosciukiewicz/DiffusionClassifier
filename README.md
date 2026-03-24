# Diffusion Classifier
This is a minimal, cleaned-up version of HCS-DFC - Diffusion Classifier for HCS repository [link](https://github.com/gmum/HCS-DFC/tree/develop)
Implementation contains example workflow for 2 digit MNIST dataset.


## Environment Setup
We utilize **UV package manager** to manage dependencies, install UV by following instructions at official website https://github.com/astral-sh/uv   

To create uv venv and install all dependencies run:
```shell
uv sync
```
and than, to activate virtual environment run
```shell
source .venv/bin/activate
```

## Training / Testing
**TODO: ADD EXAMPLE EMPTY CONFIG, ADD TESTING CODE**   
This project utilizes [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to manage training, therefore it is possible to control entire traning / testing process using configs located in `/configs` directory. 


To train backbone CNN (for two-digit MNIST only)
```shell
python lightning_training/train_cnn.py 
```
To train model run
```shell
 python scripts/train_model.py fit -c configs/[your_config_file].yaml
```
You can also override any config parameter by using [CLI arguments](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), for example:
```shell
 python scripts/train_model.py fit -c configs/[your_config_file].yaml --trainer.max_epochs=20
```
To test model
```shell
 python scripts/train_model.py test -c configs/[your_config_file].yaml
```

### Available configs
All the config files available in `/configs` directory.
- `diffusion_mnist.yaml` HCS-DFC for two-digit MNIST w/CNN backbone (requires pretrained backbone).
- `diffusion_bray.yaml` HCS-DFC for pre-extracted Bray et.al. dataset morphological features.
- `diffusion_bbbc.yaml` HCS-DFC for pre-extracted BBBC021 dataset morphological features.

## Dataset preparation
### Two digit MNIST toy dataset
### Bray et.al dataset
### BBBC021 dataset
