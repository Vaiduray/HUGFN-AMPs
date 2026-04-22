import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, EsmModel
from Hugfn_amp.tasks.base import BaseTask
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor


class PhysChemReward:
    def __init__(self, h_star=0.5, delta=0.1, lam=5.0):
        self.h_star = h_star
        self.delta = delta
        self.lam = lam

    def calculate_hydro_reward(self, sequence):
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
        reward, hydro = self.calculate_hydro_reward(sequence)
        return reward, {"Hydro": hydro}


class ESM2Classifier(nn.Module):
    def __init__(self, model_path):
        super(ESM2Classifier, self).__init__()
        self.esm = EsmModel.from_pretrained(model_path)
        self.hidden_size = self.esm.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def get_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            self.esm.eval()
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            return mean_embedding


    def predict_from_embedding(self, embedding, mc_dropout=True):
        if mc_dropout:
            self.classifier.train()
        else:
            self.classifier.eval()

        return self.classifier(embedding)

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
            kwargs must contain:
            - esm_model_path: ESM-2 local folder path
            - amp_weight_path: AMP model .pth path
            - hemo_weight_path: Hemolysis model .pth path
            - length_lambda: length penalty coefficient (default: 0.05)
        """
        obj_dim = len(objectives)
        super().__init__(tokenizer, obj_dim, max_len, transform=lambda x: x, **kwargs)

        self.objectives = objectives
        self.min_len = min_len
        self.max_len = max_len
        self.score_max = kwargs.get("score_max", [1.0] * obj_dim)
        self.length_lambda = kwargs.get("length_lambda", 0.05)

        self.esm_path = kwargs.get("esm_model_path", "./esm2_650m_weights")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(self.esm_path, local_files_only=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[AMPTask] Loading models on {self.device}...")

        self.amp_model = ESM2Classifier(self.esm_path).to(self.device)
        amp_weights = kwargs.get("amp_weight_path")
        self.amp_model.load_state_dict(torch.load(amp_weights, map_location=self.device))
        self.amp_model.eval()

        self.hemo_model = ESM2Classifier(self.esm_path).to(self.device)
        hemo_weights = kwargs.get("hemo_weight_path")
        self.hemo_model.load_state_dict(torch.load(hemo_weights, map_location=self.device))
        self.hemo_model.eval()

        self.toxi_model = ESM2Classifier(self.esm_path).to(self.device)
        toxi_weights = kwargs.get("toxi_weight_path")
        self.toxi_model.load_state_dict(torch.load(toxi_weights, map_location=self.device))
        self.toxi_model.eval()

        self.physchem_reward = PhysChemReward()

        self.length_counts = {
            10:271, 11:210, 12:274, 13:257, 14:204,
            15:312, 16:178, 17:254, 18:385, 19:156,
            20:468, 21:227, 22:188, 23:157, 24:269,
            25:207, 26:154, 27:111, 28:118, 29:152, 30:138
        }
        total = sum(self.length_counts.values())
        self.p_len = {L: c / total for L, c in self.length_counts.items()}

        max_p_len = max(self.p_len.values())
        self.p_len_normalized = {L: c / max_p_len for L, c in self.p_len.items()}

        self.EPS = 1e-8
        self.BETA = 0.6

        print("[AMPTask] Models loaded successfully.")

    def task_setup(self, *args, **kwargs):
        return [], []

    def score(self, candidates):
        """
        Args
        ----
        candidates : list of strings
            Generated sequence list, e.g. ['AKK...', 'MKL...']

        Returns
        -------
        scores : np.array
            Shape: [n_candidates, n_objectives]
            Range: [0, 1]
        """
        scores_dict = self.compute_rewards(candidates, objectives=self.objectives)

        scores = [scores_dict[obj] for obj in self.objectives]

        scores = np.stack(scores, axis=-1).astype(np.float64)

        return scores


    def compute_rewards(self, sequences, objectives):
        batch_size = 256
        num_seqs = len(sequences)

        USE_MC_DROPOUT = False

        MC_SAMPLES = 50
        BETA = 5
        SCORE_MAX = 0.99

        amp_scores = np.zeros(num_seqs)
        safety_scores = np.zeros(num_seqs)
        toxi_scores = np.zeros(num_seqs)
        len_scores = np.zeros(num_seqs)
        physchem_scores = np.zeros(num_seqs)
        length_penalty_scores = np.zeros(num_seqs)

        if "activity" in objectives or "safety" in objectives:
            for i in range(0, num_seqs, batch_size):
                batch_seqs = sequences[i : i + batch_size].tolist()

                encoding = self.esm_tokenizer(
                    batch_seqs, add_special_tokens=True, max_length=60,
                    padding=True, truncation=True, return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                with torch.no_grad():
                    if "activity" in objectives:
                        emb_amp = self.amp_model.get_embedding(input_ids, attention_mask)

                        if USE_MC_DROPOUT:
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                pred = self.amp_model.predict_from_embedding(emb_amp, mc_dropout=True)
                                preds_list.append(pred)

                            preds_stack = torch.stack(preds_list)

                            mu = preds_stack.mean(dim=0).squeeze()
                            std = preds_stack.std(dim=0).squeeze()

                            final_score = mu - BETA * std
                        else:
                            final_score = self.amp_model.predict_from_embedding(emb_amp, mc_dropout=False).squeeze()

                        final_score = torch.clamp(final_score, 0.1, SCORE_MAX)

                        amp_scores[i : i + batch_size] = final_score.cpu().numpy()

                    if "safety" in objectives:
                        emb_hemo = self.hemo_model.get_embedding(input_ids, attention_mask)

                        if USE_MC_DROPOUT:
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                pred = self.hemo_model.predict_from_embedding(emb_hemo, mc_dropout=True)
                                preds_list.append(pred)

                            preds_stack = torch.stack(preds_list)

                            mu_toxic = preds_stack.mean(dim=0).squeeze()
                            std_toxic = preds_stack.std(dim=0).squeeze()

                            penalized_toxic = mu_toxic + BETA * std_toxic
                        else:
                            mu_toxic = self.hemo_model.predict_from_embedding(emb_hemo, mc_dropout=False).squeeze()
                            penalized_toxic = mu_toxic

                        final_safety = 1.0 - penalized_toxic

                        final_safety = torch.clamp(final_safety, 0.1, SCORE_MAX)

                        safety_scores[i : i + batch_size] = final_safety.cpu().numpy()

                    if "toxi" in objectives:
                        emb_toxi = self.toxi_model.get_embedding(input_ids, attention_mask)

                        if USE_MC_DROPOUT:
                            preds_list = []
                            for _ in range(MC_SAMPLES):
                                pred = self.toxi_model.predict_from_embedding(emb_toxi, mc_dropout=True)
                                preds_list.append(pred)

                            preds_stack = torch.stack(preds_list)

                            mu_toxi = preds_stack.mean(dim=0).squeeze()
                            std_toxi = preds_stack.std(dim=0).squeeze()

                            penalized_toxi = mu_toxi + BETA * std_toxi
                        else:
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

        if "physchem" in objectives:
            for i, seq in enumerate(sequences):
                reward, details = self.physchem_reward.get_combined_reward(seq)
                physchem_scores[i] = reward

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
