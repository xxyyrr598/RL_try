import torch
import torch.nn as nn
import re
import json
import logging

logger = logging.getLogger(__name__)  # __name__ == 'reward_model'


class RewardModel(nn.Module):
    '''
    由于尝试GRPO或GSPO，因此奖励模型模仿deepseek，直接规则奖励
    '''
    def __init__(self,mode='deepseek'):
        super().__init__()
        self.mode = mode

        if mode == 'deepseek':
            # deepseek的奖励有思考奖励和正确回答奖励
            self.think_reward = 0.1
            self.answer_reward = 1
        elif mode == 'structured_generation':
            self.structure_reward = 1
    
    def forward(self, input_texts, correct_texts):
        if not isinstance(input_texts, list):
            input_texts = [input_texts]
        if not isinstance(correct_texts, list):
            correct_texts = [correct_texts]

        if len(input_texts) != len(correct_texts):
            raise ValueError("input_texts and correct_texts must have the same length")
        rewards = []
        if self.mode == 'deepseek':
            for input_text, correct_text in zip(input_texts, correct_texts):
                reward = 0
                # 思考格式奖励
                if '<think>' in input_text:
                    reward += self.think_reward
                # 回答正确奖励
                if correct_text in input_text:
                    reward += self.answer_reward
                rewards.append(reward)
        elif self.mode == 'structured_generation':
            for input_text, correct_text in zip(input_texts, correct_texts):
                reward = 0
                # 格式奖励
                try:
                    data = json.loads(input_text)
                    reward += self.structure_reward
                except Exception as e:
                    logger.error(f"JSON decode error: {e}")
                rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

