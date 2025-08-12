import torch
import torch.nn as nn
import torch.nn.functional as F
from reward_model import RewardModel

class Grpo_config:
    batch_size = 8 # 每次采样使用几个prmopt
    sample_size = 6 # 每个prompt采样几个样本
    beta_KL = 0.1 # KL散度的系数


    def __repr__(cls):
        return f"{cls.__name__}({', '.join(f'{k}={v}' for k, v in cls.__dict__.items() if not k.startswith('_'))})"



class Grpo_trainer:
    def __init__(self,model,grpo_config):
        super().__init__()
        self.policy_model = model
        self.ref_model = model
        self.reward_model = RewardModel(mode='structured_generation')
        self.grpo_config = grpo_config

    
        self.policy_model.train()
        self.ref_model.eval()
    
    def rollout(self,input_texts):
        '''
        对输入的prompt进行一次采样，得到一个batch的数据
        '''

