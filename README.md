# HUGFN for Sequence Tasks

## Installation
Tested with Python 3.9 and CUDA 11.7.
       'torch==2.0.0',
        'botorch==0.8.4',
        'hydra-core==1.3.2',
        'wandb',
        'matplotlib',
        'polyleven',
        'pymoo==0.5.0',
        'tqdm',
        'cachetools',
        'cvxopt==1.3.0',
        'plotly'
## Commands


python main.py   task=amp   algorithm=offline_mogfn   tokenizer=protein  algorithm.train_steps=10000   algorithm.batch_size=256   algorithm.eval_freq=2000   algorithm.num_samples=128  exp_name=amp  seed=4234  algorithm.state_save_path="/home/gml/GFN/mo_gfn/data/amp.pkl.gz" algorithm.sample_beta=16


