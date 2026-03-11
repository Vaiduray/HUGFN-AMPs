import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
import numpy as np
from tqdm import tqdm


HEMO_WEIGHT_PATH = "/home/gml/GFN/AMP_reward/hemolytic_classifier_650M_best.pth"
TOXI_WEIGHT_PATH = "/home/gml/GFN/AMP_reward/toxi_classifier_650M_best.pth"
ESM_MODEL_PATH = "/home/gml/GFN/AMP_reward/esm2_650m_weights"


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

    def predict_from_embedding(self, embedding):
        self.classifier.eval()
        return self.classifier(embedding)

    def forward(self, input_ids, attention_mask):
        emb = self.get_embedding(input_ids, attention_mask)
        return self.predict_from_embedding(emb)


class PeptidePredictor:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading models on {self.device}...")
        
        self.esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
        
        self.hemo_model = ESM2Classifier(ESM_MODEL_PATH).to(self.device)
        self.hemo_model.load_state_dict(torch.load(HEMO_WEIGHT_PATH, map_location=self.device))
        self.hemo_model.eval()
        
        self.toxi_model = ESM2Classifier(ESM_MODEL_PATH).to(self.device)
        self.toxi_model.load_state_dict(torch.load(TOXI_WEIGHT_PATH, map_location=self.device))
        self.toxi_model.eval()
        
        print("Models loaded successfully.")

    def predict_batch(self, sequences):
        """
        批量预测毒性和溶血活性
        
        Args:
            sequences: 序列列表
            
        Returns:
            dict: 包含 'toxicity_prob' 和 'hemolysis_prob' 的字典
        """
        if not sequences:
            return {'toxicity_prob': [], 'hemolysis_prob': []}
        
        encoding = self.esm_tokenizer(
            sequences, add_special_tokens=True, max_length=60,
            padding=True, truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            emb_hemo = self.hemo_model.get_embedding(input_ids, attention_mask)
            emb_toxi = self.toxi_model.get_embedding(input_ids, attention_mask)
            
            hemo_probs = self.hemo_model.predict_from_embedding(emb_hemo).squeeze().cpu().numpy()
            toxi_probs = self.toxi_model.predict_from_embedding(emb_toxi).squeeze().cpu().numpy()
            
            if hemo_probs.ndim == 0:
                hemo_probs = np.array([hemo_probs.item()])
                toxi_probs = np.array([toxi_probs.item()])
        
        return {'toxicity_prob': toxi_probs, 'hemolysis_prob': hemo_probs}

    def predict_csv(self, input_csv, output_csv, sequence_col='sequence', batch_size=256):
        """
        读取CSV文件，预测毒性和溶血活性，并保存结果
        
        Args:
            input_csv: 输入CSV文件路径
            output_csv: 输出CSV文件路径
            sequence_col: 序列列名
            batch_size: 批次大小（控制显存使用）
        """
        print(f"Reading CSV file: {input_csv}")
        df = pd.read_csv(input_csv)
        
        if sequence_col not in df.columns:
            raise ValueError(f"Column '{sequence_col}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        sequences = df[sequence_col].tolist()
        num_seqs = len(sequences)
        
        print(f"Total sequences: {num_seqs}")
        print(f"Batch size: {batch_size}")
        
        all_toxi_probs = []
        all_hemo_probs = []
        
        for i in tqdm(range(0, num_seqs, batch_size), desc="Predicting"):
            batch_seqs = sequences[i:i + batch_size]
            results = self.predict_batch(batch_seqs)
            all_toxi_probs.extend(results['toxicity_prob'])
            all_hemo_probs.extend(results['hemolysis_prob'])
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        df['toxicity_prob'] = all_toxi_probs
        df['hemolysis_prob'] = all_hemo_probs
        
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
        print(f"\nSummary statistics:")
        print(f"Toxicity probability - Mean: {np.mean(all_toxi_probs):.4f}, Std: {np.std(all_toxi_probs):.4f}")
        print(f"Hemolysis probability - Mean: {np.mean(all_hemo_probs):.4f}, Std: {np.std(all_hemo_probs):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Predict toxicity and hemolysis for peptide sequences')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--sequence_col', type=str, default='Sequence', 
                        help='Column name containing sequences')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size for prediction (reduce if OOM)')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cuda/cpu), default auto-detect')
    
    args = parser.parse_args()
    
    predictor = PeptidePredictor(device=args.device)
    
    predictor.predict_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        sequence_col=args.sequence_col,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
