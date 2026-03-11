import hydra
import wandb
import math
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F

from torch_seq_moo.algorithms.base import BaseAlgorithm
from torch_seq_moo.algorithms.mogfn_utils.utils import mean_pairwise_distances, generate_simplex, thermometer, plot_pareto, pareto_frontier
from torch_seq_moo.utils import str_to_tokens, tokens_to_str
from torch_seq_moo.metrics import get_all_metrics

from torch.distributions import Categorical
from tqdm import tqdm

_SMOOTHED_GEO_EPSILON = 0.1


def smoothed_geometric_mean(rewards, weights, epsilon=_SMOOTHED_GEO_EPSILON):
    """
    带平滑的加权几何平均聚合函数
    
    Args:
        rewards: 奖励向量，形状为 [n_objectives] 或 [batch_size, n_objectives]
        weights: 偏好权重，形状为 [n_objectives]
        epsilon: 平滑因子，直接写死防止 log(0)
    
    Returns:
        聚合后的标量奖励
    """
    rewards = torch.clamp(rewards, min=0.0)
    log_rewards = (rewards + epsilon).log()
    weighted_log_sum = (weights * log_rewards).sum(dim=-1)
    return weighted_log_sum.exp()


class MOGFNSubTB(BaseAlgorithm):
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        super(MOGFNSubTB, self).__init__(cfg, task, tokenizer, task_cfg)
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        self.obj_dim = self.task.obj_dim
        
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
        self.eval_prefs = self.get_eval_prefs()
        
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
        
        self.cfg.model.vocab_size = len(self.tokenizer.full_vocab)
        self.cfg.model.num_actions = len(self.tokenizer.non_special_vocab) + 1
        
        self.subtb = cfg.subtb
        self.short_lengths = self.subtb.short_lengths
        self.random_long_ratio = self.subtb.random_long_ratio
        self.min_subtraj_len = self.subtb.min_subtraj_len
        self.max_subtraj_len = self.subtb.max_subtraj_len
        self.max_subtraj_per_traj = self.subtb.max_subtraj_per_traj
        self.include_terminal = self.subtb.include_terminal
        self.weight_scheme = self.subtb.weight_scheme
        self.lambda_weight = self.subtb.lambda_weight

    def get_eval_prefs(self):
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

    def optimize(self, task, init_data=None):
        losses, rewards = [], []
        hv, r2, hsri, rs = 0., 0., 0., np.zeros(self.obj_dim)
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} HV: {:.3f} R2: {:.3f} HSRI: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(rs.mean(), hv, r2, hsri, sum(losses[-10:]) / 10 if losses else 0, sum(rewards[-10:]) / 10 if rewards else 0))

        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            if i != 0 and i % self.eval_freq == 0:
                with torch.no_grad():
                    samples, all_rews, rs, mo_metrics, topk_metrics, fig = self.evaluation(task, plot=True)
                hv, r2, hsri = mo_metrics["hypervolume"], mo_metrics["r2"], mo_metrics["hsri"]
                
                self.log(dict(
                    topk_rewards=topk_metrics[0].mean(),
                    topk_diversity=topk_metrics[1].mean(),
                    sample_r=rs.mean()
                ), commit=False)
                
                if self.use_eval_pref:
                    self.log({"topk_reward_pref_{}".format(idx): val for idx, val in enumerate(topk_metrics[0])}, commit=False)
                    self.log({"topk_diversity_pref_{}".format(idx): val for idx, val in enumerate(topk_metrics[1])}, commit=False)
                    self.log({"sample_reward_pref_{}".format(idx): val for idx, val in enumerate(rs)}, commit=False)

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
                
                if i % self.pareto_freq == 0:
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
        cond_var, (prefs, beta) = self._get_condition_var(train=True, bs=batch_size)
        
        states, _ = self.sample(batch_size, cond_var)
        
        if isinstance(prefs, np.ndarray) and prefs.ndim == 1:
            prefs = np.tile(prefs, (batch_size, 1))
        
        log_r = self.process_reward(states, prefs, task).to(self.device)
        
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        
        subtrajectories = self.sample_subtrajectories()
        loss = self.compute_subtb_loss(subtrajectories, log_r)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), log_r.mean()

    def sample(self, episodes, cond_var=None, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)
        step_logprobs = []
        state_flows = []
        
        if cond_var is None:
            cond_var, _ = self._get_condition_var(train=train, bs=episodes)
        
        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()[:1]
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)
        
        for t in (range(self.max_len) if episodes > 0 else []):
            logits_all, log_F_all = self.model(x, cond_var, lens=lens, mask=None, return_all=True, return_flow=True)
            logits = logits_all[-1]
            log_F = log_F_all[-1]
            
            if t <= self.min_len:
                logits[:, 0] = -1000
                if t == 0:
                    traj_logprob += self.model.Z(cond_var)

            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            policy_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                num_actions = logits.shape[-1]
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), num_actions, (episodes, )).to(self.device), actions)
            
            log_prob = policy_dist.log_prob(actions) * active_mask
            step_logprobs.append(log_prob)
            state_flows.append(log_F.squeeze(-1))
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        
        states = tokens_to_str(x.t(), self.tokenizer)
        traj_lengths = lens + t + 1 if t > 0 else lens + 1
        
        step_logprobs_tensor = torch.stack(step_logprobs) if step_logprobs else torch.zeros(self.max_len, episodes).to(self.device)
        state_flows_tensor = torch.stack(state_flows) if state_flows else torch.zeros(self.max_len, episodes).to(self.device)
        
        if train:
            self._last_step_logprobs = step_logprobs_tensor
            self._last_state_flows = state_flows_tensor
            self._last_lengths = traj_lengths
            self._last_cond_var = cond_var
        
        return states, traj_logprob

    def sample_subtrajectories(self):
        batch_size = self._last_lengths.size(0)
        all_subtrajectories = []
        
        for b in range(batch_size):
            traj_len = self._last_lengths[b].item()
            subtrajs = []
            
            short_count = int(self.max_subtraj_per_traj * (1 - self.random_long_ratio))
            long_count = self.max_subtraj_per_traj - short_count
            
            for length in self.short_lengths:
                for start in range(traj_len - length):
                    subtrajs.append((start, start + length))
                    if len(subtrajs) >= short_count:
                        break
                if len(subtrajs) >= short_count:
                    break
            
            if len(subtrajs) < short_count:
                subtrajs = subtrajs[:short_count]
            
            long_lengths = list(range(self.min_subtraj_len, min(self.max_subtraj_len + 1, traj_len)))
            if long_lengths:
                for _ in range(long_count):
                    if not long_lengths:
                        break
                    length = random.choice(long_lengths)
                    start = random.randint(0, traj_len - length)
                    subtrajs.append((start, start + length))
            
            if self.include_terminal:
                for length in self.short_lengths:
                    if traj_len > length:
                        subtrajs.append((traj_len - length, traj_len))
            
            all_subtrajectories.append(subtrajs)
        
        return all_subtrajectories

    def get_subtraj_weight(self, length):
        if self.weight_scheme == "geometric":
            return self.lambda_weight ** length
        elif self.weight_scheme == "length_based":
            if length <= 3:
                return 1.0
            elif length <= 6:
                return 0.7
            elif length <= 10:
                return 0.4
            else:
                return 0.25
        return 1.0

    def compute_subtb_loss(self, subtrajectories, log_r):
        total_loss = 0.0
        total_weight = 0.0
        
        batch_size = self._last_lengths.size(0)
        cond_var = self._last_cond_var
        
        log_Z = self.model.Z(cond_var)
        
        for b in range(batch_size):
            step_logprobs = self._last_step_logprobs[:, b]
            state_flows = self._last_state_flows[:, b]
            traj_len = self._last_lengths[b].item()
            
            subtrajs = subtrajectories[b]
            
            if len(subtrajs) == 0:
                continue
            
            subtraj_weights = []
            subtraj_losses = []
            
            for start, end in subtrajs:
                length = end - start
                
                sub_logprob = step_logprobs[start:end].sum()
                
                if start == 0:
                    log_F_start = log_Z[b]
                else:
                    log_F_start = state_flows[start]
                
                if end == traj_len:
                    log_F_end = log_r[b]
                else:
                    log_F_end = state_flows[end]
                
                flow_balance = log_F_start + sub_logprob - log_F_end
                subtraj_loss = flow_balance.pow(2)
                
                weight = self.get_subtraj_weight(length)
                
                subtraj_weights.append(weight)
                subtraj_losses.append(subtraj_loss * weight)
                total_weight += weight
            
            if subtraj_losses:
                batch_loss = torch.stack(subtraj_losses).mean()
                total_loss += batch_loss
        
        if total_weight > 0:
            total_loss = total_loss / batch_size
        else:
            total_loss = torch.tensor(0.0).to(self.device)
        
        return total_loss

    def process_reward(self, seqs, prefs, task, rewards=None, train=True):
        if rewards is None:
            rewards = task.score(seqs)
        
        rewards_tensor = torch.tensor(rewards)
        num_seqs = len(seqs)
        log_r_list = []
        
        if isinstance(prefs, np.ndarray) and prefs.ndim == 1:
            prefs = np.tile(prefs, (num_seqs, 1))
        elif isinstance(prefs, list) and not isinstance(prefs[0], (list, np.ndarray)):
            prefs = [prefs]
        
        for i, pref in enumerate(prefs):
            obj_dim = len(pref)
            if rewards.shape[1] == obj_dim + 1:
                task_reward = (torch.tensor(pref) * rewards_tensor[i, :obj_dim]).sum()
                length_penalty = rewards_tensor[i, -1].item()
                log_r = task_reward + length_penalty
                log_r = max(log_r, self.reward_min)
            else:
                if self.reward_type == "convex":
                    log_r = (torch.tensor(pref) * rewards_tensor[i]).sum().clamp(min=self.reward_min).log()
                elif self.reward_type == "logconvex":
                    log_r = (torch.tensor(pref) * rewards_tensor[i].clamp(min=self.reward_min).log()).sum()
                elif self.reward_type == "tchebycheff":
                    log_r = (torch.tensor(pref) * torch.abs(1 - rewards_tensor[i])).max().clamp(min=self.reward_min).log()
                elif self.reward_type == "smoothed_geo":
                    log_r = smoothed_geometric_mean(rewards_tensor[i], torch.tensor(pref)).clamp(min=self.reward_min).log()
                log_r = log_r.item()
            
            log_r_list.append(log_r)
        
        log_r = torch.tensor(log_r_list)
        return log_r.exp() if not train else log_r

    def evaluation(self, task, plot=False):
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
                
                topk_r, topk_idx = torch.topk(r, self.k)
                samples_arr = np.array(samples)
                topk_seq = samples_arr[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
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
                
                topk_r, topk_idx = torch.topk(r, self.k)
                samples_arr = np.array(samples)
                topk_seq = samples_arr[topk_idx].tolist()
                edit_dist = mean_pairwise_distances(topk_seq)
                topk_rs.append(topk_r.mean().item())
                topk_div.append(edit_dist)
                
                max_idx = r.argmax()
                new_candidates.append(samples[max_idx])
                all_rewards.append(rewards[max_idx])
                r_scores.append(r.max().item())

        r_scores = np.array(r_scores)
        all_rewards = np.array(all_rewards)
        new_candidates = np.array(new_candidates)

        if not self.use_eval_pref:
            pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
            
            mo_metrics = get_all_metrics(pareto_targets, self.eval_metrics, hv_ref=self._ref_point, r2_prefs=self.simplex, num_obj=self.obj_dim)
            obj_names = self.task_cfg.objectives if hasattr(self.task_cfg, "objectives") else [f"obj_{i}" for i in range(self.obj_dim)]
            fig = plot_pareto(pareto_targets, all_rewards, pareto_only=False, objective_names=obj_names) if plot else None        
        else:
            mo_metrics = {met: 0 for met in self.eval_metrics}
            fig = None
        
        return new_candidates, all_rewards, r_scores, mo_metrics, (np.array(topk_rs), np.array(topk_div)), fig

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
        pareto_candidates, pareto_targets = pareto_frontier(new_candidates, all_rewards, maximize=True)
        return new_candidates, all_rewards, r_scores, pareto_candidates, pareto_targets, prefs

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

        if self.pref_use_therm:
            prefs_enc = thermometer(torch.from_numpy(prefs), self.therm_n_bins, 0, 1) 
        else: 
            prefs_enc = torch.from_numpy(prefs)
        
        if self.beta_use_therm:
            beta_enc = thermometer(torch.from_numpy(np.array([beta])), self.therm_n_bins, 0, self.beta_max) 
        else:
            beta_enc = torch.from_numpy(np.array([beta]))
        
        if self.beta_cond:
            cond_var = torch.cat((prefs_enc.view(-1), beta_enc.view(-1))).float().to(self.device)
        else:
            cond_var = prefs_enc.view(-1).float().to(self.device)
        
        if bs:
            cond_var = torch.tile(cond_var.unsqueeze(0), (bs, 1))
        
        return cond_var, (prefs, beta)
