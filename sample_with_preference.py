import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

def parse_prefs(pref_str):
    if pref_str is None:
        return None
    
    try:
        if pref_str.startswith('[') and pref_str.endswith(']'):
            prefs = json.loads(pref_str)
        else:
            prefs = [float(x.strip()) for x in pref_str.split(',')]
        prefs = np.array(prefs, dtype=np.float64)
        prefs = prefs / prefs.sum()
        return prefs
    except Exception as e:
        print(f"Error parsing preferences: {e}")
        return None

def find_available_runs(data_dir="data"):
    runs = []
    for item in os.listdir(data_dir):
        run_dir = os.path.join(data_dir, item)
        if os.path.isdir(run_dir):
            config_file = os.path.join(run_dir, "hydra_config.txt")
            model_file = None
            for f in os.listdir(run_dir):
                if f.endswith(".pkl.gz"):
                    model_file = os.path.join(run_dir, f)
                    break
            if model_file and os.path.exists(config_file):
                runs.append({
                    'name': item,
                    'config': config_file,
                    'model': model_file
                })
    return runs

def load_config_from_yaml(config_path):
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    config_dict = yaml.safe_load(config_content)
    return config_dict

def instantiate_config(config_dict, task_name="amp"):
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create(config_dict)
    
    tokenizer_cfg = cfg.get('tokenizer', {'_target_': 'torch_seq_moo.utils.ResidueTokenizer'})
    task_cfg = cfg.get('task', None)
    algorithm_cfg = cfg.get('algorithm', None)
    
    return cfg, tokenizer_cfg, task_cfg, algorithm_cfg

def get_preference_encoding(prefs, algorithm):
    from Hugfn_amp.algorithms.mogfn_utils.utils import thermometer
    
    device = next(algorithm.model.parameters()).device
    
    prefs = np.array(prefs)
    
    if prefs.ndim == 1:
        if algorithm.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), algorithm.therm_n_bins, 0, 1)
        else:
            prefs_enc = torch.from_numpy(prefs)
        prefs_enc = prefs_enc.view(-1).float().to(device)
    else:
        if algorithm.pref_use_therm:
            prefs_enc = torch.stack([
                thermometer(torch.from_numpy(p), algorithm.therm_n_bins, 0, 1)
                for p in prefs
            ])
        else:
            prefs_enc = torch.from_numpy(prefs)
        prefs_enc = prefs_enc.view(prefs.shape[0], -1).float().to(device)
    
    if algorithm.beta_use_therm:
        beta_enc = thermometer(torch.from_numpy(np.array([algorithm.sample_beta])), 
                              algorithm.therm_n_bins, 0, algorithm.beta_max)
    else:
        beta_enc = torch.from_numpy(np.array([algorithm.sample_beta]))
    
    if algorithm.beta_cond:
        cond_var = torch.cat((prefs_enc, beta_enc.repeat(prefs_enc.shape[0], 1)), dim=1)
    else:
        cond_var = prefs_enc
    
    return cond_var

def sample_sequences(algorithm, num_samples, prefs, batch_size=256):
    from Hugfn_amp.utils import str_to_tokens, tokens_to_str
    from torch.distributions import Categorical
    
    device = next(algorithm.model.parameters()).device
    model = algorithm.model
    max_len = algorithm.max_len
    min_len = algorithm.min_len
    sampling_temp = algorithm.sampling_temp
    random_action_prob = algorithm.random_action_prob
    
    cond_var = get_preference_encoding(prefs, algorithm)
    
    all_states = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        if prefs.ndim == 1:
            batch_cond_var = cond_var.unsqueeze(0).repeat(current_batch_size, 1)
        else:
            batch_cond_var = cond_var[batch_idx * batch_size : (batch_idx + 1) * current_batch_size]
        
        states = [''] * current_batch_size
        traj_logprob = torch.zeros(current_batch_size).to(device)
        active_mask = torch.ones(current_batch_size).bool().to(device)
        x = str_to_tokens(states, algorithm.tokenizer).to(device).t()[:1]
        lens = torch.zeros(current_batch_size).long().to(device)
        uniform_pol = torch.empty(current_batch_size).fill_(random_action_prob).to(device)
        
        for t in range(max_len):
            logits = model(x, batch_cond_var, lens=lens, mask=None)
            
            if t <= min_len:
                logits[:, 0] = -1000
                if t == 0:
                    traj_logprob += model.Z(batch_cond_var)
            else:
                logits[:, 0] += 10
            
            sampling_dist = Categorical(logits=logits / sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            
            if random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(
                    uniform_mix, 
                    torch.randint(int(t <= min_len), logits.shape[1], (current_batch_size, )).to(device), 
                    actions
                )
            
            log_prob = policy_dist.log_prob(actions) * active_mask
            traj_logprob += log_prob
            
            actions_apply = torch.where(
                torch.logical_not(active_mask), 
                torch.zeros(current_batch_size).to(device).long(), 
                actions + 4
            )
            active_mask = torch.where(active_mask, actions != 0, active_mask)
            
            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        
        batch_states = tokens_to_str(x.t(), algorithm.tokenizer)
        all_states.extend(batch_states)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Sampled: {(batch_idx + 1) * batch_size}/{num_samples}")
    
    return all_states

def main():
    parser = argparse.ArgumentParser(description='MOGFN Preference Sampling')
    parser.add_argument('--task', type=str, default='amp_10',
                       help='Task name (e.g., amp_10, amp_12, amp_13)')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of sequences to sample')
    parser.add_argument('--prefs', type=str, default=None,
                       help='Preference vector (JSON array or comma-separated, e.g., "[0.5,0.3,0.2]")')
    parser.add_argument('--output', type=str, default='sampling_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode: prompt for preferences')
    parser.add_argument('--list_runs', action='store_true',
                       help='List all available runs')
    
    args = parser.parse_args()
    
    if args.list_runs:
        runs = find_available_runs()
        print("Available runs:")
        for run in runs:
            print(f"  - {run['name']}")
            print(f"      Config: {run['config']}")
            print(f"      Model:  {run['model']}")
        return
    
    task_name = args.task
    data_dir = "data"
    run_dir = os.path.join(data_dir, task_name)
    
    config_file = os.path.join(run_dir, "hydra_config.txt")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        print("\nAvailable runs:")
        runs = find_available_runs()
        for run in runs:
            print(f"  - {run['name']}")
        return
    
    model_file = None
    for f in os.listdir(run_dir):
        if f.endswith(".pkl.gz"):
            model_file = os.path.join(run_dir, f)
            break
    
    if not model_file:
        print(f"Error: No model file found in {run_dir}")
        return
    
    print(f"Loading config from: {config_file}")
    config_dict = load_config_from_yaml(config_file)
    
    tokenizer_cfg = config_dict.get('tokenizer', {'_target_': 'torch_seq_moo.utils.ResidueTokenizer'})
    task_cfg = config_dict.get('task', None)
    algorithm_cfg = config_dict.get('algorithm', None)
    
    print("Initializing components...")
    import hydra
    from hydra.utils import instantiate
    
    tokenizer = instantiate(tokenizer_cfg)
    print(f"Tokenizer initialized with vocab size: {len(tokenizer.full_vocab)}")
    
    task = instantiate(task_cfg, tokenizer=tokenizer)
    print(f"Task initialized: {task_cfg.get('_target_', 'Unknown')}")
    
    algorithm = instantiate(algorithm_cfg, task=task, tokenizer=tokenizer, 
                           cfg=algorithm_cfg, task_cfg=task_cfg)
    print("Algorithm initialized")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    algorithm.model.to(device)
    
    print(f"Loading model weights from: {model_file}")
    import gzip
    import pickle
    try:
        with gzip.open(model_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model_state = checkpoint.get('model_state', None)
        if model_state is not None:
            algorithm.model.load_state_dict(model_state)
            print("Model weights loaded successfully!")
        else:
            print("Warning: No model state found in checkpoint")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    prefs = None
    if args.interactive:
        print("\n" + "=" * 60)
        print("MOGFN Preference Sampling Interface")
        print("=" * 60)
        print(f"Objective dimensions: {algorithm.obj_dim}")
        
        obj_names = task_cfg.get('objectives', [f'obj_{i}' for i in range(algorithm.obj_dim)])
        print(f"Objective names: {obj_names}")
        print("\nPlease input your preference vector (comma-separated or JSON array)")
        print("Example: 0.5,0.3,0.2")
        print("Or: [0.5,0.3,0.2]")
        print("Values should sum to 1.0")
        
        while True:
            pref_input = input("\nEnter preferences: ").strip()
            prefs = parse_prefs(pref_input)
            if prefs is not None and len(prefs) == algorithm.obj_dim:
                break
            elif prefs is not None:
                print(f"Error: Expected {algorithm.obj_dim} preferences, got {len(prefs)}")
            else:
                print("Invalid input format. Please try again.")
    elif args.prefs:
        prefs = parse_prefs(args.prefs)
        if prefs is None or len(prefs) != algorithm.obj_dim:
            print(f"Error: Invalid preference format. Expected {algorithm.obj_dim} values.")
            return
    else:
        print("Error: Please specify preferences using --prefs or use --interactive for interactive mode")
        return
    
    num_samples = args.num_samples
    print(f"\nPreferences: {prefs}")
    print(f"Num samples: {num_samples}")
    print("Sampling...")
    
    with torch.no_grad():
        samples = sample_sequences(algorithm, num_samples, prefs, batch_size=256)
    
    print("Evaluating...")
    samples_array = np.array(samples)
    rewards = task.score(samples_array)
    
    obj_names = task_cfg.get('objectives', [f'obj_{i}' for i in range(algorithm.obj_dim)])
    prefs_tensor = torch.tensor(prefs).to(device)
    rewards_tensor = torch.tensor(rewards).to(device)
    
    if algorithm.reward_type == "convex":
        combined_rewards = (prefs_tensor * rewards_tensor).sum(axis=1).cpu().numpy()
    elif algorithm.reward_type == "logconvex":
        combined_rewards = (prefs_tensor * rewards_tensor.clamp(min=algorithm.reward_min).log()).sum(axis=1).exp().cpu().numpy()
    elif algorithm.reward_type == "tchebycheff":
        combined_rewards = (prefs_tensor * torch.abs(1 - rewards_tensor)).max(axis=1)[0].cpu().numpy()
    else:
        combined_rewards = (prefs_tensor * rewards_tensor).sum(axis=1).cpu().numpy()
    
    results_df = pd.DataFrame({
        'sequence': samples,
        'combined_reward': combined_rewards
    })
    
    for i, obj_name in enumerate(obj_names):
        results_df[obj_name] = rewards[:, i]
    
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Preferences: {prefs}")
    print(f"Total samples: {len(samples)}")
    print(f"\nMean rewards:")
    for i, obj_name in enumerate(obj_names):
        print(f"  {obj_name}: {rewards[:, i].mean():.4f}")
    print(f"  Combined: {combined_rewards.mean():.4f}")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
