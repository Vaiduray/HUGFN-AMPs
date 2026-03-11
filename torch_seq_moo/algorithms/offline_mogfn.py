import hydra
import wandb
import math
import time
import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F

from torch_seq_moo.algorithms.base import BaseAlgorithm
from torch_seq_moo.algorithms.mogfn_utils.utils import mean_pairwise_distances, generate_simplex, thermometer, plot_pareto, pareto_frontier
from torch_seq_moo.algorithms.mogfn_subtb import smoothed_geometric_mean
from torch_seq_moo.utils import str_to_tokens, tokens_to_str
from torch_seq_moo.metrics import get_all_metrics

from torch.distributions import Categorical
from tqdm import tqdm



class MOGFN(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(MOGFN, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        # GFN stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.therm_n_bins = cfg.therm_n_bins
        self.beta_use_therm = cfg.beta_use_therm
        self.pref_use_therm = cfg.pref_use_therm
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.beta_cond = cfg.beta_cond
        self.pref_cond = cfg.pref_cond
        self.beta_scale = cfg.beta_scale
        self.beta_shape = cfg.beta_shape
        self.pref_alpha = cfg.pref_alpha
        self.beta_max = cfg.beta_max
        self.reward_type = cfg.reward_type
        self.use_eval_pref = cfg.use_eval_pref
        self.num_pareto_points = cfg.num_pareto_points
        self.state_save_path = cfg.state_save_path
        self.pareto_freq = cfg.pareto_freq
        self.eval_prefs = self.get_eval_pref()# np.array(self.task_cfg.eval_pref)

        # Offline data config
        self.use_offline = getattr(cfg, 'use_offline', False)
        self.offline_data_path = getattr(cfg, 'offline_data_path', None)
        self.offline_batch_size = getattr(cfg, 'offline_batch_size', 16)
        
        # New evaluation config
        self.use_new_eval = getattr(cfg, 'use_new_eval', False)
        self.new_eval_num_samples = getattr(cfg, 'new_eval_num_samples', 5000)
        self.new_eval_save_path = getattr(cfg, 'new_eval_save_path', "results/new_eval_sequences.csv")
        
        # Eval Stuff
        self._hv_ref = None
        self._ref_point = np.array([0] * self.obj_dim)
        self.eval_metrics = cfg.eval_metrics
        self.eval_freq = cfg.eval_freq
        self.k = cfg.k
        self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.simplex = generate_simplex(self.obj_dim, cfg.simplex_bins)
        self.unnormalize_rewards = cfg.unnormalize_rewards
        
        # Adapt model config to task
        self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1

    def get_eval_pref(self):
        rs = np.random.RandomState(123)
        return rs.dirichlet([1] * self.obj_dim, size=5)

    def init_policy(self):
        cfg = self.cfg
        pref_dim = self.therm_n_bins * self.obj_dim if self.pref_use_therm else self.obj_dim
        beta_dim = self.therm_n_bins if self.beta_use_therm else 1
        cond_dim = pref_dim + beta_dim if self.beta_cond else pref_dim
        self.model = hydra.utils.instantiate(cfg.model, cond_dim=cond_dim, use_cond=(self.beta_cond or self.pref_cond))

        self.model.to(self.device)
        
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        
        self.load_offline_data()

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.obj_dim)
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} HV: {:.3f} R2: {:.3f} HSRI: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))

        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(task, plot=True)
                
                if self.use_new_eval:
                    hv, r2, hsri = 0.0, 0.0, 0.0
                    topk_metrics = None
                    self.log({f"new_eval_{k}": v for k, v in mo_metrics.items()}, commit=False)
                    if fig is not None:
                        self.log(dict(new_eval_fig=fig), commit=False)
                else:
                    hv, r2, hsri = mo_metrics["hypervolume"], mo_metrics["r2"], mo_metrics["hsri"]
                    self.log(dict(
                        topk_rewards=topk_metrics[0].mean(),
                        topk_diversity=topk_metrics[1].mean(),
                        sample_r=rs.mean()
                    ), commit=False)
                    if self.use_eval_pref:
                        self.log({"topk_reward_pref_{}".format(i): topk_metrics[0][i] for i in range(len(topk_metrics[0]))}, commit=False)
                        self.log({"topk_diversity_pref_{}".format(i): topk_metrics[1][i] for i in range(len(topk_metrics[1]))}, commit=False)
                        self.log({"sample_reward_pref_{}".format(i): rs[i] for i in range(len(rs))}, commit=False)

                    self.log({key: val for key, val in mo_metrics.items()}, commit=False)

                    if fig is not None:
                        self.log(dict(
                            pareto_front=fig
                        ), commit=False)
                    table = wandb.Table(columns = ["Sequence", "Rewards", "Prefs"])
                    if self.unnormalize_rewards:
                        all_rews *= task.score_max
                    for sample, rew, pref in zip(samples, all_rews, self.simplex):
                        table.add_data(str(sample), str(rew), str(pref))
                    self.log({"generated_seqs": table})
                
                if i % self.pareto_freq == 0 and not self.use_new_eval:
                    new_candidates, all_rewards, r_scores, pareto_candidates, pareto_targets, prefs = self.plot_pareto(self.num_pareto_points, task)
                    self.update_state(dict(
                        topk_rewards=topk_metrics[0].mean(),
                        topk_diversity=topk_metrics[1].mean(),
                        sample_r=rs.mean(),
                        hv=mo_metrics["hypervolume"],
                        r2=mo_metrics["r2"],
                        samples=new_candidates,
                        all_rewards=all_rewards,
                        prefs=prefs
                    ))
                    self.save_state()
            self.log(dict(
                train_loss=loss,
                train_rewards=r,
            ))
            pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        
        return {
            'losses': losses,
            'train_rs': rewards,
            'hypervol_rel': hv
        }
    
    def train_step(self, task, batch_size):
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        if self.use_offline and self.offline_sequences is not None:
            return self.hybrid_train_step(task, batch_size)
        else:
            cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
            states, logprobs = self.sample(batch_size, cond_var)
            log_r = self.process_reward(states, prefs, task).to(self.device)
            loss = (logprobs - beta * log_r).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
            self.opt.step()
            self.opt_Z.step()
            return loss.item(), log_r.mean()
    
    def hybrid_train_step(self, task, batch_size):
        online_bs = batch_size - self.offline_batch_size
        
        if online_bs < 0:
            raise ValueError(f"offline_batch_size ({self.offline_batch_size}) must be <= batch_size ({batch_size})")
        
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        
        if online_bs > 0:
            online_cond = cond_var[:online_bs]
            online_states, online_logprobs = self.sample(online_bs, online_cond)
            online_rewards = self.process_reward(online_states, prefs, task).to(self.device)
        else:
            online_states, online_logprobs = [], torch.tensor([]).to(self.device)
            online_rewards = torch.tensor([]).to(self.device)
        
        if self.offline_batch_size > 0 and self.offline_sequences is not None:
            offline_idx = np.random.choice(len(self.offline_sequences), self.offline_batch_size, replace=False)
            offline_states = [self.offline_sequences[i] for i in offline_idx]
            offline_rewards_raw = self.offline_rewards[offline_idx]
            
            offline_rewards_tensor = torch.tensor(offline_rewards_raw).to(self.device)
            offline_prefs_tensor = torch.tensor(np.tile(prefs, (self.offline_batch_size, 1))).to(self.device)
            offline_rewards = self._process_offline_reward(offline_rewards_tensor, offline_prefs_tensor)
            
            offline_cond = cond_var if self.offline_batch_size == batch_size else cond_var[online_bs:]
            offline_logprobs = self._get_log_prob(offline_states, offline_cond)
        else:
            offline_states, offline_logprobs = [], torch.tensor([]).to(self.device)
            offline_rewards = torch.tensor([]).to(self.device)
        
        combined_logprobs = torch.cat([online_logprobs, offline_logprobs]) if len(online_logprobs) > 0 and len(offline_logprobs) > 0 else online_logprobs if len(online_logprobs) > 0 else offline_logprobs
        combined_log_r = torch.cat([online_rewards, offline_rewards]) if len(online_rewards) > 0 and len(offline_rewards) > 0 else online_rewards if len(online_rewards) > 0 else offline_rewards
        
        if combined_logprobs.numel() > 0:
            beta_tensor = torch.tensor([beta] * len(combined_logprobs)).to(self.device) if isinstance(beta, (int, float)) else beta
            loss = (combined_logprobs - beta_tensor * combined_log_r).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
            self.opt.step()
            self.opt_Z.step()
            
            combined_rewards = online_rewards if len(online_rewards) > 0 else offline_rewards
            return loss.item(), combined_rewards.mean() if combined_rewards.numel() > 0 else torch.tensor(0.0).to(self.device)
        else:
            return 0.0, torch.tensor(0.0).to(self.device)
    
    def _process_offline_reward(self, rewards_tensor, prefs_tensor):
        if self.reward_type == "convex":
            log_r = (prefs_tensor * rewards_tensor).sum(axis=1).clamp(min=self.reward_min).log()
        elif self.reward_type == "logconvex":
            log_r = (prefs_tensor * rewards_tensor.clamp(min=self.reward_min).log()).sum(axis=1)
        elif self.reward_type == "tchebycheff":
            log_r = (prefs_tensor * torch.abs(1 - rewards_tensor)).max(axis=1)[0].clamp(min=self.reward_min).log()
        elif self.reward_type == "smoothed_geo":
            log_r = smoothed_geometric_mean(rewards_tensor, prefs_tensor).clamp(min=self.reward_min).log()
        return log_r


    def sample(self, episodes, cond_var=None, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1]
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            logits = self.model(x, cond_var, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z(cond_var)
            else:
                logits[:, 0] += 0 # Force stopping after min length

            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)
            
            log_prob = policy_dist.log_prob(actions) * active_mask
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        return states, traj_logprob
    
    def process_reward(self, seqs, prefs, task, rewards=None, train=True):
        if rewards is None:
            rewards = task.score(seqs)
        
        rewards_tensor = torch.tensor(rewards).to(self.device)
        prefs_tensor = torch.tensor(prefs).to(self.device)
        
        if self.reward_type == "convex":
            log_r = (prefs_tensor * rewards_tensor).sum(axis=1).clamp(min=self.reward_min).log()
        elif self.reward_type == "logconvex":
            log_r = (prefs_tensor * rewards_tensor.clamp(min=self.reward_min).log()).sum(axis=1)
        elif self.reward_type == "tchebycheff":
            log_r = (prefs_tensor * torch.abs(1 - rewards_tensor)).max(axis=1)[0].clamp(min=self.reward_min).log()
        elif self.reward_type == "smoothed_geo":
            log_r = smoothed_geometric_mean(rewards_tensor, prefs_tensor).clamp(min=self.reward_min).log()
        
        return log_r.exp() if not train else log_r

    def evaluation(self, task, plot=False):
        if self.use_new_eval:
            return self.new_evaluation(task)
        
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        topk_rs = []
        topk_div = []
        if self.use_eval_pref:
            for prefs in self.eval_prefs:
                cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
                samples, _ = self.sample(self.num_samples, cond_var, train=False)
                rewards = task.score(samples)
                r = self.process_reward(samples, prefs, task, rewards=rewards, train=False)
                
                # topk metrics
                topk_r, topk_idx = torch.topk(r, self.k)
                if isinstance(samples, torch.Tensor):
                    samples = samples.cpu().numpy()
                if isinstance(topk_idx, torch.Tensor):
                    topk_idx = topk_idx.cpu().numpy()
                topk_seq = samples[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
                # top 1 metrics
                max_idx = r.argmax()
                new_candidates.append(samples[max_idx])
                all_rewards.append(rewards[max_idx])
                r_scores.append(r.max().item())
        else:
            for prefs in self.simplex:
                cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=self.num_samples)
                samples, _ = self.sample(self.num_samples, cond_var, train=False)
                rewards = task.score(samples)
                r = self.process_reward(samples, prefs, task, rewards=rewards, train=False)
                
                # topk metrics
                topk_r, topk_idx = torch.topk(r, self.k)
                if isinstance(samples, torch.Tensor):
                    samples = samples.cpu().numpy()
                if isinstance(topk_idx, torch.Tensor):
                    topk_idx = topk_idx.cpu().numpy()
                topk_seq = samples[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
                # top 1 metrics
                max_idx = r.argmax()
                if isinstance(max_idx, torch.Tensor):
                    max_idx = max_idx.cpu().numpy()
                new_candidates.append(samples[max_idx])
                all_rewards.append(rewards[max_idx])
                r_scores.append(r.max().item())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)

        if not self.use_eval_pref:
            # filter to get current pareto front 
            pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
            
            mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, r2_prefs=self.simplex, num_obj=self.obj_dim)
            obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in range(self.obj_dim)]
            fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None        
        else:
            mo_metrics = {met: 0 for met in self.eval_metrics}
            fig = None
        
        return new_candidates, all_rewards, r_scores, mo_metrics, (np.array(topk_rs), np.array(topk_div)), fig
    
    def new_evaluation(self, task):
        import os
        import pandas as pd
        
        obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else ['activity', 'hemo', 'toxi']
        
        uniform_prefs = np.random.dirichlet([1] * self.obj_dim, size=self.new_eval_num_samples)
        
        all_samples = []
        all_rewards = []
        
        batch_size = 256
        num_batches = (self.new_eval_num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.new_eval_num_samples)
            current_batch_size = end_idx - start_idx
            
            prefs = uniform_prefs[start_idx:end_idx]
            cond_var, (_, beta) = self._get_condition_var(prefs=prefs, train=False, bs=current_batch_size)
            samples, _ = self.sample(current_batch_size, cond_var, train=False)
            rewards = task.score(samples)
            
            all_samples.extend(samples)
            all_rewards.extend(rewards)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"New eval sampling: {end_idx}/{self.new_eval_num_samples}")
        
        all_rewards = np.array(all_rewards)
        
        os.makedirs(os.path.dirname(self.new_eval_save_path) if os.path.dirname(self.new_eval_save_path) else '.', exist_ok=True)
        results_df = pd.DataFrame({
            'sequence': all_samples,
            **{obj_names[i]: all_rewards[:, i] for i in range(self.obj_dim)}
        })
        results_df.to_csv(self.new_eval_save_path, index=False)
        print(f"Saved {len(all_samples)} sequences to {self.new_eval_save_path}")
        
        reward_means = all_rewards.mean(axis=0)
        reward_stds = all_rewards.std(axis=0)
        
        seq_lengths = np.array([len(seq) for seq in all_samples])
        length_mean = seq_lengths.mean()
        length_std = seq_lengths.std()
        
        chm_c, chm_h, chm_hm = 0, 0, 0
        
        new_eval_metrics = {
            f'{obj_names[0]}_mean': float(reward_means[0]),
            f'{obj_names[1]}_mean': float(reward_means[1]),
            f'{obj_names[2]}_mean': float(reward_means[2]),
            f'{obj_names[0]}_std': float(reward_stds[0]),
            f'{obj_names[1]}_std': float(reward_stds[1]),
            f'{obj_names[2]}_std': float(reward_stds[2]),
            'seq_length_mean': float(length_mean),
            'seq_length_std': float(length_std),
            'C_ratio': float(chm_c),
            'H_ratio': float(chm_h),
            'HM_ratio': float(chm_hm),
            'num_samples': len(all_samples)
        }
        
        rs = np.random.rand(len(all_samples))
        return all_samples, all_rewards, rs, new_eval_metrics, None, None

    def plot_pareto(self, num_points, task):
        prefs = np.random.dirichlet([1] * self.obj_dim, size=num_points)
        new_candidates = []
        r_scores = [] 
        all_rewards = []
        for pref in prefs:
            cond_var, (_, beta) = self._get_condition_var(prefs=pref, train=False, bs=self.num_samples)
            samples, _ = self.sample(self.num_samples, cond_var, train=False)
            rewards = task.score(samples)
            r = self.process_reward(samples, pref, task, rewards=rewards, train=False)
            
            new_candidates.extend(samples)
            all_rewards.extend(rewards)
            r_scores.extend(r.cpu())
        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards,maximize=True)
        return new_candidates, all_rewards, r_scores, pareto_candidates, pareto_targets, prefs

    def val_step(self, batch_size):
        overall_loss = 0.
        for pref in self.simplex:
            cond_var, (prefs, beta) = self._get_condition_var(prefs=pref, train=False, bs=batch_size)
            num_batches = len(self.val_split.inputs) // self.batch_size
            losses = 0
            for i in range(num_batches):
                states = self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size]
                logprobs = self._get_log_prob(states, cond_var, batch_cond=None)
                r = self.process_reward(self.val_split.inputs[i * self.batch_size:(i+1) * self.batch_size], prefs).to(seq_logits.device)
                loss = (seq_logits - beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

                losses += loss.item()
            overall_loss += (losses / num_batches)
        return overall_loss / len(self.simplex)

    def _get_log_prob(self, states, cond_var, batch_cond=None):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        mask = x.eq(self.tokenizer.padding_idx)
        logits = self.model(x, cond_var, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, 21)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z(cond_var)
        return seq_logits

    def _get_condition_var(self, prefs=None, beta=None, train=True, bs=None):
        if prefs is None:
            if not train:
                prefs = self.simplex[0]
            else:
                prefs = np.random.dirichlet(np.array(self.pref_alpha))
        
        if beta is None:
            if train:
                beta = float(np.random.randint(1, self.beta_max+1)) if self.beta_cond else self.sample_beta
            else:
                beta = self.sample_beta
        
        prefs = np.array(prefs)
        
        if prefs.ndim == 1:
            if self.pref_use_therm:
                prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1)
            else:
                prefs_enc = torch.from_numpy(prefs)
            prefs_enc = prefs_enc.view(-1).float().to(self.device)
        else:
            if self.pref_use_therm:
                prefs_enc = torch.stack([
                    thermometer(torch.from_numpy(p), self.therm_n_bins, 0, 1)
                    for p in prefs
                ])
            else:
                prefs_enc = torch.from_numpy(prefs)
            prefs_enc = prefs_enc.view(prefs.shape[0], -1).float().to(self.device)
        
        if self.beta_use_therm:
            beta_enc = thermometer(torch.from_numpy(np.array([beta])), self.therm_n_bins, 0, self.beta_max)
        else:
            beta_enc = torch.from_numpy(np.array([beta]))
        
        if self.beta_cond:
            cond_var = torch.cat((prefs_enc, beta_enc.repeat(prefs_enc.shape[0], 1)), dim=1)
        else:
            cond_var = prefs_enc
        
        if bs and prefs.ndim == 1:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        
        return cond_var, (prefs, beta)
    
    def load_offline_data(self):
        if not self.use_offline or self.offline_data_path is None:
            self.offline_sequences = None
            self.offline_rewards = None
            return
        
        import os
        if not os.path.exists(self.offline_data_path):
            print(f"Warning: Offline data path does not exist: {self.offline_data_path}")
            self.offline_sequences = None
            self.offline_rewards = None
            return
        
        df = pd.read_csv(self.offline_data_path)
        self.offline_sequences = df['sequence'].tolist()
        
        reward_cols = ['activity', 'hemo', 'toxi']
        missing_cols = [col for col in reward_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing reward columns: {missing_cols}")
            reward_cols = [col for col in reward_cols if col in df.columns]
        
        self.offline_rewards = df[reward_cols].values.astype(np.float32)
        
        print(f"Loaded {len(self.offline_sequences)} offline sequences from {self.offline_data_path}")
        print(f"Reward dimensions: {self.offline_rewards.shape[1]}")
        print(f"Sample sequence: {self.offline_sequences[0] if self.offline_sequences else 'None'}")
        print(f"Sample rewards: {self.offline_rewards[0] if len(self.offline_rewards) > 0 else 'None'}")
