# HUGFN-AMPs for Antimicrobial Peptide Generation

Code for antimicrobial peptide generation with Hybrid Uncertainty-aware Generative Flow Networks. 

This folder consists of code for AMP generation:

- **AMP Design** (Objectives: antimicrobial activity, hemolysis safety, toxicity)

## Installation

The code is encapsulated in the `HUgfn_amp` library. To install the library along with the dependencies follow the instructions below. Tested with Python 3.9 and CUDA 12.8.

```bash
pip install -r requirements.txt
```


## Commands

For AMP generation:

```bash
python main.py \
    task=amp \
    algorithm=offline_mogfn \
    tokenizer=protein \
    algorithm.train_steps=10000 \
    algorithm.batch_size=256 \
    algorithm.eval_freq=2000 \
    algorithm.num_samples=128 \
    exp_name=amp \
    seed=123 \
    algorithm.state_save_path="./data/amp.pkl.gz" \
    algorithm.sample_beta=16
```
