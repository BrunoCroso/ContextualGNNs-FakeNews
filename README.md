# Enriching GNNs with Contextual Text Representations for Fake News Detection on Social Media

[[Paper]](https://arxiv.org/pdf/2410.19193)

This repository builds on the work of [nkanak/detection-of-fake-news-campaigns](https://github.com/nkanak/detection-of-fake-news-campaigns) and introduces enhancements to improve the performance of fake news detection models. The improvements include refined architectures, optimized training strategies, and better utilization of graph and text-based features. Below is a detailed guide for setting up and running the models.

---

## Table of Contents
1. [Setup](#setup)
2. [Usage](#usage)
3. [Key Commands](#key-commands)
4. [Citing This Work](#citing-this-work)

---

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/thomas-ferraz/improvement-of-fake-news-detection-models.git
    cd improvement-of-fake-news-detection-models
    ```

2. **Install dependencies**:  
   Ensure Python 3.8 or above is installed. Then, run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download necessary datasets**:  
   Place the datasets in the appropriate directory (e.g., `data/`).

4. **Preprocess the datasets**:  
   Preprocess text and graph features as required:
    ```bash
    python preprocess.py --config configs/preprocess_config.yaml
    ```

---

## Usage

1. **Train a model**:  
   To train a model using default configurations, run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

2. **Evaluate a model**:  
   To evaluate a trained model on a test set, use:
    ```bash
    python evaluate.py --config configs/evaluate_config.yaml
    ```

3. **Run experiments with multiple configurations**:  
   You can launch experiments for different model configurations by modifying and using:
    ```bash
    python experiment_runner.py --config configs/experiment_config.yaml
    ```

---

## Key Commands

| **Command**                     | **Description**                                   |
|----------------------------------|---------------------------------------------------|
| `python preprocess.py`           | Preprocess input datasets.                       |
| `python train.py`                | Train a fake news detection model.               |
| `python evaluate.py`             | Evaluate the trained model on a test dataset.    |
| `python experiment_runner.py`    | Run multiple experiments with various settings.  |

---

## Citing This Work

If you use this code or find it helpful in your research, please cite it (bibtex): 

```
@inproceedings{
silva2024enriching,
title={Enriching {GNN}s with Text Contextual Representations for Detecting Disinformation Campaigns on Social Media},
author={Bruno Croso Cunha da Silva and Thomas Palmeira Ferraz and Roseli De Deus Lopes},
booktitle={The Third Learning on Graphs Conference},
year={2024},
url={https://openreview.net/forum?id=2jAcyVTtbB}
}
```

Please also consider citing our base work, in which this codebase is inspired and extends:

```
@article{michail2022detection,
  title={Detection of fake news campaigns using graph convolutional networks},
  author={Michail, Dimitrios and Kanakaris, Nikos and Varlamis, Iraklis},
  journal={International Journal of Information Management Data Insights},
  volume={2},
  number={2},
  pages={100104},
  year={2022},
  publisher={Elsevier}
}
```


