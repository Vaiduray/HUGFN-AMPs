import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, EsmModel
from torch_seq_moo.tasks.base import BaseTask
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor


class PhysChemReward:
    def __init__(self, h_star=0.5, delta=0.1, lam=5.0):
        """
        疏水性区间奖励计算器
        h_star: 目标中心值 (default: 0.5)
        delta: 容忍带宽 (default: 0.1), 区间为 [h_star - delta, h_star + delta] = [0.4, 0.6]
        lam: 惩罚强度系数 (default: 5.0)
        """
        self.h_star = h_star
        self.delta = delta
        self.lam = lam

    def calculate_hydro_reward(self, sequence):
        """
        使用 modlamp 计算疏水性，并应用区间奖励
        目标: 疏水值越接近 [0.4, 0.6] 区间越好
        """
        try:
            desc = PeptideDescriptor(sequence, 'eisenberg')
            desc.calculate_hydrophobicity()
            hydro = desc.descriptor[0][0]

            dist = abs(hydro - self.h_star)
            excess = max(0, dist - self.delta)
            reward = math.exp(-self.lam * excess)

            return reward, hydro

        except Exception:
            return 0.0, 0.0

    def get_combined_reward(self, sequence):
        """
        获取疏水性奖励
        """
        reward, hydro = self.calculate_hydro_reward(sequence)
        return reward, {"Hydro": hydro}


# ==============================================================================
# 1. 模型结构定义 这里是使用了 Mean Pooling 版本
# ==============================================================================
class ESM2Classifier(nn.Module):
    def __init__(self, model_path):
        super(ESM2Classifier, self).__init__()
        # 加载基础模型
        self.esm = EsmModel.from_pretrained(model_path)
        self.hidden_size = self.esm.config.hidden_size
        
        # 分类头 (保持和你训练时完全一致)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), # 关键：这里有 Dropout
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), # 关键：这里有 Dropout
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    # 新增函数：只跑 ESM 提取特征
    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad(): # ESM 部分永远不需要梯度，也不需要 Dropout
            self.esm.eval() 
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # Mean Pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            return mean_embedding

    # 新增函数：只跑 MLP 分类
    def predict_from_embedding(self, embedding, mc_dropout=True):
        if mc_dropout:
            self.classifier.train() # 开启 Dropout！
        else:
            self.classifier.eval()  # 关闭 Dropout
            
        return self.classifier(embedding)

    # 保持原 forward 不变，兼容旧代码
    def forward(self, input_ids, attention_mask):
        emb = self.get_embedding(input_ids, attention_mask)
        return self.predict_from_embedding(emb, mc_dropout=True)

# ==============================================================================
# 2. GFlowNet 任务类 (AMPTask)
# ==============================================================================
class AMPTask(BaseTask):
    def __init__(
        self,
        tokenizer,
        objectives,
        max_len,
        min_len,
        **kwargs
    ):
        """
        Args:
            kwargs 必须包含:
            - esm_model_path: ESM-2 本地文件夹路径 (用于加载 Tokenizer 和 Base Model)
            - amp_weight_path: 抗菌模型 .pth 路径
            - hemo_weight_path: 溶血模型 .pth 路径
            - length_lambda: 长度惩罚系数 λ (default: 0.05)
        """
        obj_dim = len(objectives)
        super().__init__(tokenizer, obj_dim, max_len, transform=lambda x: x, **kwargs)
        
        self.objectives = objectives
        self.min_len = min_len
        self.max_len = max_len
        self.score_max = kwargs.get("score_max", [1.0] * obj_dim)
        self.length_lambda = kwargs.get("length_lambda", 0.05)
        
        # ---------------- 1. 加载 ESM 专用 Tokenizer ----------------
        self.esm_path = kwargs.get("esm_model_path", "./esm2_650m_weights")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_path, local_files_only=True)
        
        # ---------------- 2. 加载预训练模型权重 ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[AMPTask] Loading models on {self.device}...")
        
        # A. 加载抗菌模型
        self.amp_model = ESM2Classifier(self.esm_path).to(self.device)
        amp_weights = kwargs.get("amp_weight_path")
        self.amp_model.load_state_dict(torch.load(amp_weights, map_location=self.device))
        self.amp_model.eval() # 开启评估模式
    
        # B. 加载溶血模型
        self.hemo_model = ESM2Classifier(self.esm_path).to(self.device)
        hemo_weights = kwargs.get("hemo_weight_path")
        self.hemo_model.load_state_dict(torch.load(hemo_weights, map_location=self.device))
        self.hemo_model.eval()

        # C. 加载毒性模型
        self.toxi_model = ESM2Classifier(self.esm_path).to(self.device)
        toxi_weights = kwargs.get("toxi_weight_path")
        self.toxi_model.load_state_dict(torch.load(toxi_weights, map_location=self.device))
        self.toxi_model.eval()
        
        # D. 初始化理化性质奖励计算器
        self.physchem_reward = PhysChemReward()
        
        # E. 初始化长度先验分布
        self.length_counts = {
            10:271, 11:210, 12:274, 13:257, 14:204,
            15:312, 16:178, 17:254, 18:385, 19:156,
            20:468, 21:227, 22:188, 23:157, 24:269,
            25:207, 26:154, 27:111, 28:118, 29:152, 30:138
        }
        total = sum(self.length_counts.values())
        self.p_len = {L: c / total for L, c in self.length_counts.items()}
        
        # 归一化：让最常见的长度奖励接近 1
        max_p_len = max(self.p_len.values())
        self.p_len_normalized = {L: c / max_p_len for L, c in self.p_len.items()}
        
        self.EPS = 1e-8
        self.BETA = 0.6
        
        print("[AMPTask] Models loaded successfully.")

    def task_setup(self, *args, **kwargs):
        # 可以在这里返回一些初始种子序列，如果没有就返回空
        return [], []

    def score(self, candidates):
        """
        Args
        ----
        candidates : list of strings
            生成的序列列表，例如 ['AKK...', 'MKL...']

        Returns
        -------
        scores : np.array
            Shape: [n_candidates, n_objectives]
            值域: [0, 1]
        """
        # 1. 计算所有目标的原始分数字典
        scores_dict = self.compute_rewards(candidates, objectives=self.objectives)
        
        # 2. 按 objectives 列表的顺序提取分数
        scores = [scores_dict[obj] for obj in self.objectives]
        
        # 3. 堆叠成矩阵
        scores = np.stack(scores, axis=-1).astype(np.float64)
        
        # 注意：这里的 scores 已经是 [0,1] 的概率了，不需要再像 Nupack 那样除以 score_max
        # 也不需要乘以 -1，因为我们的目标是最大化这三个值
        return scores


    def compute_rewards(self, sequences, objectives):
        batch_size = 256  # 增大batch_size以提高GPU利用率
        num_seqs = len(sequences)
        
        # ==================== MC Dropout 开关 ====================
        # True: 使用 MC Dropout 计算不确定性惩罚 (Mean - BETA * Std)
        # False: 直接使用模型预测的均值，不考虑不确定性
        USE_MC_DROPOUT = False
        # ==========================================================
        
        # 定义 MC Dropout 的采样次数和惩罚系数
        MC_SAMPLES = 50  # 采样次数：越多越稳定，但耗时
        BETA = 5      # 惩罚系数 (Mean - BETA * Std)
        SCORE_MAX = 0.99  # 硬上限：分数不会超过此值
        
        # 初始化
        amp_scores = np.zeros(num_seqs)
        safety_scores = np.zeros(num_seqs)
        toxi_scores = np.zeros(num_seqs)
        len_scores = np.zeros(num_seqs)
        physchem_scores = np.zeros(num_seqs)
        length_penalty_scores = np.zeros(num_seqs)
        
        if "activity" in objectives or "safety" in objectives:
            for i in range(0, num_seqs, batch_size):
                batch_seqs = sequences[i : i + batch_size].tolist()
                
                # 1. Tokenize
                encoding = self.esm_tokenizer(
                    batch_seqs, add_special_tokens=True, max_length=60,
                    padding=True, truncation=True, return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad(): # 全程不需要梯度
                    # -------------------------------------------------------
                    # 优化核心：先用 ESM 提取特征 (Embedding)
                    # 这个步骤最耗时，但对同一批数据我们只跑一次！
                    # -------------------------------------------------------
                    # 注意：我们需要分别提取两个模型的特征，因为它们是两个独立微调的模型
                    
                    if "activity" in objectives:
                        # 拿到 amp_model 的特征
                        emb_amp = self.amp_model.get_embedding(input_ids, attention_mask)
                        
                        if USE_MC_DROPOUT:
                            # MC Dropout 采样循环
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                # 注意：传入 mc_dropout=True
                                pred = self.amp_model.predict_from_embedding(emb_amp, mc_dropout=True)
                                preds_list.append(pred) # [batch, 1]
                            
                            # 堆叠 -> [MC_SAMPLES, batch, 1]
                            preds_stack = torch.stack(preds_list)
                            
                            # 计算统计量
                            mu = preds_stack.mean(dim=0).squeeze() # [batch]
                            std = preds_stack.std(dim=0).squeeze()   # [batch]
                            
                            # 计算惩罚后的奖励 (带硬上限)
                            final_score = mu - BETA * std
                        else:
                            # 直接使用模型预测，不考虑不确定性
                            final_score = self.amp_model.predict_from_embedding(emb_amp, mc_dropout=False).squeeze()
                        
                        final_score = torch.clamp(final_score, 0.1, SCORE_MAX)
                        
                        amp_scores[i : i + batch_size] = final_score.cpu().numpy()

                    if "safety" in objectives:
                        # 拿到 hemo_model 的特征
                        emb_hemo = self.hemo_model.get_embedding(input_ids, attention_mask)
                        
                        if USE_MC_DROPOUT:
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                # 这里预测的是毒性概率
                                pred = self.hemo_model.predict_from_embedding(emb_hemo, mc_dropout=True)
                                preds_list.append(pred)
                            
                            preds_stack = torch.stack(preds_list)
                            
                            # 毒性概率的均值和标准差
                            mu_toxic = preds_stack.mean(dim=0).squeeze()
                            std_toxic = preds_stack.std(dim=0).squeeze()
                            
                            # 这里的逻辑稍微绕一点：
                            # 我们希望"安全"，即毒性低。
                            # 如果模型对"毒性"很不确定(std大)，说明这个序列很危险/未知。
                            # 原始奖励: Safety = 1 - Toxic
                            # 悲观估计毒性 -> 毒性 = mu_toxic + BETA * std_toxic
                            # Safety = 1 - (mu_toxic + BETA * std_toxic)
                            
                            penalized_toxic = mu_toxic + BETA * std_toxic
                        else:
                            # 直接使用模型预测
                            mu_toxic = self.hemo_model.predict_from_embedding(emb_hemo, mc_dropout=False).squeeze()
                            penalized_toxic = mu_toxic
                        
                        final_safety = 1.0 - penalized_toxic
                        
                        final_safety = torch.clamp(final_safety, 0.1, SCORE_MAX)
                        
                        safety_scores[i : i + batch_size] = final_safety.cpu().numpy()

                    if "toxi" in objectives:
                        # 拿到 toxi_model 的特征
                        emb_toxi = self.toxi_model.get_embedding(input_ids, attention_mask)
                        
                        if USE_MC_DROPOUT:
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                pred = self.toxi_model.predict_from_embedding(emb_toxi, mc_dropout=True)
                                preds_list.append(pred)
                            
                            preds_stack = torch.stack(preds_list)
                            
                            mu_toxi = preds_stack.mean(dim=0).squeeze()
                            std_toxi = preds_stack.std(dim=0).squeeze()
                            
                            # 毒性越低越好，和safety逻辑相同
                            penalized_toxi = mu_toxi + BETA * std_toxi
                        else:
                            # 直接使用模型预测
                            mu_toxi = self.toxi_model.predict_from_embedding(emb_toxi, mc_dropout=False).squeeze()
                            penalized_toxi = mu_toxi
        
                        final_toxi = 1.0 - penalized_toxi
                        
                        final_toxi = torch.clamp(final_toxi, 0.1, SCORE_MAX)
                        
                        toxi_scores[i : i + batch_size] = final_toxi.cpu().numpy()

        if "length" in objectives:
            for i, seq in enumerate(sequences):
                L = len(seq)
                if L == 0:
                    len_scores[i] = 0.0
                    continue
                
                if L < 10 or L > 30:
                    len_scores[i] = 0.0
                else:
                    len_scores[i] = (self.p_len_normalized[L] + self.EPS) ** self.BETA

        # ---------------- 计算理化性质奖励 ----------------
        if "physchem" in objectives:
            for i, seq in enumerate(sequences):
                reward, details = self.physchem_reward.get_combined_reward(seq)
                physchem_scores[i] = reward

        # ---------------- 组装返回字典 ----------------
        dict_return = {}
        
        if "activity" in objectives:
            dict_return["activity"] = amp_scores
            
        if "safety" in objectives:
            dict_return["safety"] = safety_scores

        if "toxi" in objectives:
            dict_return["toxi"] = toxi_scores
            
        if "length" in objectives:
            dict_return["length"] = len_scores
            
        if "physchem" in objectives:
            dict_return["physchem"] = physchem_scores

        if "length_penalty" in objectives:
            dict_return["length_penalty"] = length_penalty_scores

        return dict_return



