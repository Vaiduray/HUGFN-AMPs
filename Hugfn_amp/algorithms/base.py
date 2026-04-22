import wandb
import pickle
import gzip
import torch

class BaseAlgorithm():
    def __init__(self, cfg, task, tokenizer, task_cfg, **kwargs):
        self.cfg = cfg
        self.task = task
        self.tokenizer = tokenizer
        self.task_cfg = task_cfg
        self.state = {}

    def optimize(self, task, initial_data=None):
        raise NotImplementedError("Override this method in your class")
    
    def log(self, metrics, commit=True):
        wandb.log(metrics, commit=True)
    
    def update_state(self, metrics):
        for k, v in metrics.items():
            if k in self.state.keys():
                self.state[k].append(v)
            else:
                self.state[k] = [v]

    def get_model_state_dict(self):
        """返回模型权重的字典。子类应重写此方法以返回正确的模型状态。"""
        if hasattr(self, 'model'):
            return self.model.state_dict()
        return None

    def save_state(self):
        save_dict = {
            'state': self.state,
            'model_state': self.get_model_state_dict(),
        }
        with gzip.open(self.state_save_path, 'wb+') as f:
            pickle.dump(save_dict, f)
