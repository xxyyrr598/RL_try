import torch
import torch.nn as nn
import torch.nn.functional as F


class reference_model:
    '''
    ref model类，用来接收policy model生成的结果并计算logits
    用来算KL散度
    '''
    def __init__(self,
                 model,
                 tokenizer,
                 device='cuda'):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = tokenizer
    

    def get_logits_with_mask(self, input_texts):
        '''
        输入 policy model 生成的文本，输出 ref model 的 logits 和 response_mask。
        
        输入：
            input_texts: List[Dict], 格式为 [{"prompt": "...", "response": "..."}, ...]
        
        输出：
            all_logits: (B, L_full, V) - ref model 的原始 logits
            response_mask: (B, L_full) - bool tensor，True 表示该 logits 位置对应 response token
                        （注意：使用时需与 logits 一起裁剪为 :-1）
        '''
        if not isinstance(input_texts, list):
            raise ValueError("input_texts must be a list")
        if len(input_texts) == 0:
            raise ValueError("input_texts cannot be empty")

        # 拼接 prompt + response
        full_input_texts = [item['prompt'] + item['response'] for item in input_texts]

        # Tokenize 完整文本（带 padding）
        full_inputs = self.tokenizer(
            full_input_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)

        # Tokenize prompt（用于计算 prompt 长度）
        prompt_texts = [item['prompt'] for item in input_texts]
        prompt_inputs = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**full_inputs)
            all_logits = outputs.logits  # (B, L_full, V)
            B, L_full, V = all_logits.shape

        # 获取每个样本的真实长度（去掉 padding）
        prompt_lengths = prompt_inputs['attention_mask'].sum(dim=1).tolist()      # [p_len1, p_len2, ...]

        # 创建 response_mask: (B, L_full)
        response_mask = torch.zeros(B, L_full, dtype=torch.bool, device=all_logits.device)

        for i, plen in enumerate(prompt_lengths):
            start_idx = plen - 1  # logits[t] 预测 input_ids[t+1]，所以 response 从 plen-1 开始
            if start_idx >= 0 and start_idx < L_full:
                response_mask[i, start_idx:] = True
            # 注意：不需要手动设末尾为 False，后续用 :-1 裁剪

        return all_logits, response_mask, full_inputs

    def get_log_probs(self, input_texts):
        """
        计算每个样本中「response 部分」的 token-level log-probability
        
        输入：
            input_texts: List[Dict], 格式为 [{"prompt": "...", "response": "..."}, ...]
        
        输出：
            log_probs_list: List[torch.Tensor]，每个元素是一个 1D tensor，表示该样本 response 部分的 log-prob
                            长度等于 response token 数量
        """
        if not isinstance(input_texts, list):
            raise ValueError("input_texts must be a list")
        if len(input_texts) == 0:
            return []

        all_logits, response_mask, full_inputs = self.get_logits_with_mask(input_texts)
        # all_logits: (B, L_full, V)
        # response_mask: (B, L_full)
        # full_inputs['input_ids']: (B, L_full)

        input_ids = full_inputs['input_ids']  # (B, L_full)
        B, L_full = input_ids.shape

        # 对齐 logits 和 labels
        logits = all_logits[:, :-1, :]  # (B, L-1, V)
        labels = input_ids[:, 1:]       # (B, L-1)

        log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, V)
        token_logps = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, L-1)

        response_mask = response_mask[:, :-1]  # (B, L-1)

        # 使用 mask 保留 response 区域
        masked_logps = token_logps.masked_fill(~response_mask, 0.0)  # 非 response 设为 0

        log_probs_list = []
        for i in range(B):
            # 提取response部分
            valid_logps = masked_logps[i][response_mask[i]]  # 只取 True 的位置
            log_probs_list.append(valid_logps)

        return log_probs_list  # List[torch.Tensor], 每个 shape (L_response_i,)

        

