## Can deep learning match the efficiency of human visual long-term memory for object details? 

This repository contains the code, stimuli, pretrained models, and simulation results reported in the following paper:

Orhan AE (2022) [Can deep learning match the efficiency of human visual long-term memory to store object details?](https://arxiv.org/abs/2204.13061) arXiv:2204.13061.

Part of the code here is adapted from [Andrej Karpathy's minimalistic GPT (minGPT) implementation](https://github.com/karpathy/minGPT).

### Directories:

* [`stimuli`](https://github.com/eminorhan/igpt-memory/blob/master/stimuli/): contains the study and test images used in the simulated versions of the [Brady et al. (2008)](https://www.pnas.org/doi/abs/10.1073/pnas.0803390105) and [Konkle et al. (2010)](https://psycnet.apa.org/record/2010-15559-009) experiments.

* [`results`](https://github.com/eminorhan/igpt-memory/blob/master/results/): contains the simulation results from all experiments and code for reading and plotting the results.

* [`scripts`](https://github.com/eminorhan/igpt-memory/blob/master/scripts/): contains example SLURM scripts for running the code on an HPC cluster.

* [`mingpt`](https://github.com/eminorhan/igpt-memory/blob/master/mingpt/): contains utility functions for the iGPT model (adapted from [Andrej Karpathy's minGPT implementation](https://github.com/karpathy/minGPT)).

### Code description:

* [`train.py`](https://github.com/eminorhan/igpt-memory/blob/master/train.py): trains an iGPT model on a given dataset.

* [`finetune.py`](https://github.com/eminorhan/igpt-memory/blob/master/finetune.py): finetunes a model on the study set of a recognition memory experiment.

* [`run_random_noise_expt.py`](https://github.com/eminorhan/igpt-memory/blob/master/run_random_noise_expt.py): runs the random noise experiment reported in Figure 3c in the paper.

* [`test.py`](https://github.com/eminorhan/igpt-memory/blob/master/test.py): evaluates a model on the test set of a recognition memory experiment.

* [`generate.py`](https://github.com/eminorhan/igpt-memory/blob/master/generate.py): generates samples from an iGPT model.

### Pretrained models:

* [`iGPT-S-ImageNet.pt`](https://drive.google.com/file/d/1C83ZFk46fZFgGHo5QpdeUqLg0jToWxeH/view?usp=sharing): iGPT-S model pretrained on ImageNet (1.9 GB).

* [`iGPT-mini-ImageNet.pt`](https://drive.google.com/file/d/1XdJDgYv2e9cvd52COTjULZCxei970aeh/view?usp=sharing): iGPT-mini model pretrained on ImageNet (0.5 GB).

* [`iGPT-S-SAYCam.pt`](https://drive.google.com/file/d/1LPB7fNzuICrCw0ty40snduUrrh-HWe2m/view?usp=sharing): iGPT-S model pretrained on SAYCam (1.9 GB).

* [`iGPT-S-SAYCam-0.1.pt`](https://drive.google.com/file/d/1VaE_AIz6nla53fMJaG2UhrnMkz1TsQ4X/view?usp=sharing): iGPT-S model pretrained on 10% of SAYCam (1.9 GB).

* [`iGPT-S-SAYCam-0.01.pt`](https://drive.google.com/file/d/19Svc2SienQ56FGqNInyST8kLclCJGQeV/view?usp=sharing): iGPT-S model pretrained on 1% of SAYCam (1.9 GB).
