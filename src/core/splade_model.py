import torch
from transformers import AutoModelForMaskedLM
from typing import List, Dict, Union
from .tokenizers import SpladeTokenizer

class SpladeModel:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = SpladeTokenizer(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # CPU에서 실행 (GPU 메모리 충돌 방지)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> Dict[str, Union[List[int], List[float]]]:
        all_indices = []
        all_values = []
        
        # 배치 처리 (64개씩 처리)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                # CPU에서 실행하므로 autocast 제거
                inputs = self.tokenizer.tokenize(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                output = self.model(**inputs)
                logits = output.logits
                
                # ReLU 적용 & log 적용 & 필요없는 0 제거
                values, _ = torch.max(
                    torch.log(1 + torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), 
                    dim=1
                )
                
                # 0이 아닌 값들만 추출해서 sparse vector로 변환
                batch_values = values.numpy()
                for row in batch_values:
                    non_zero_mask = row > 0
                    indices = non_zero_mask.nonzero()[0]
                    vals = row[non_zero_mask]
                    
                    all_indices.append(indices)
                    all_values.append(vals)
                        
        return {"indices": all_indices, "values": all_values}

    def encode(self, text: str) -> Dict[int, float]:
        result = self.encode_batch([text], batch_size=1)
        indices = result["indices"][0]
        values = result["values"][0]
        return dict(zip(indices, values))
