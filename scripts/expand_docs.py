import os
import sys
import json
import torch
import ir_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from typing import List, Dict

# 상수처리
MODEL_NAME = "castorini/doc2query-t5-base-msmarco"
SUM_MODEL_NAME = "michau/t5-base-en-generate-headline"
DATASET_ID = "wikir/en1k/training"
OUTPUT_FILE = "data/expanded_docs.json"
BATCH_SIZE = 32
NUM_QUERIES = 10
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
SEED = 42

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    device = torch.device("cuda")

    # 시드 설정을 통해 재현성 보장
    torch.manual_seed(SEED)

    # 모델 로딩
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        sys.exit(1)

    # 데이터셋 로딩
    dataset = ir_datasets.load(DATASET_ID)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    expanded_documents = []
    batch_docs = []
    
    try:
        total_docs = dataset.docs_count()
    except Exception:
        total_docs = None

    pbar = tqdm(total=total_docs, unit="docs", desc="확장 중")

    # 문서 확장
    for doc in dataset.docs_iter():
        batch_docs.append(doc)
        
        if len(batch_docs) >= BATCH_SIZE:
            _process_batch(batch_docs, model, tokenizer, device, expanded_documents)
            pbar.update(len(batch_docs))
            batch_docs = []
            
    # 남은 문서 처리
    if batch_docs:
        _process_batch(batch_docs, model, tokenizer, device, expanded_documents)
        pbar.update(len(batch_docs))
    
    pbar.close()

    # 메모리 정리
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    # 문서의 제목을 michau/t5-base-en-generate-headline 모델을 이용해 생성
    try:
        sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)
        sum_model.to(device)
        sum_model.eval()
    except Exception as e:
        print(f"헤드라인 생성 모델 로딩 오류: {e}")
        sys.exit(1)

    total_expanded = len(expanded_documents)    
    pbar = tqdm(total=total_expanded, unit="docs", desc="제목 생성 중")
    
    for i in range(0, total_expanded, BATCH_SIZE):
        batch = expanded_documents[i:i + BATCH_SIZE]
        _generate_titles_batch(batch, sum_model, sum_tokenizer, device)
        pbar.update(len(batch))
        
    pbar.close()

    # 결과 저장
    # JSON 파일로 저장해서 indexing 과정에서 JSON을 읽어서 인덱싱하도록 수정
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(expanded_documents, f, indent=2, ensure_ascii=False)
    
    print("문서 확장 및 제목 생성 완료")

def _generate_titles_batch(
    docs_batch: List[Dict],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device
):
    original_texts = [d['original_text'] for d in docs_batch]
    input_texts = ["headline: " + text for text in original_texts]
    
    # 토큰화
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            summary_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=2,
                early_stopping=True
            )
            
    titles = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    # 결과 반영
    for doc, title in zip(docs_batch, titles):
        doc['title'] = title

# 속도를 위해 배치 처리
def _process_batch(
    docs_batch: List, 
    model: T5ForConditionalGeneration, 
    tokenizer: T5Tokenizer, 
    device: torch.device, 
    results_list: List[Dict]
):
    original_texts = [d.text for d in docs_batch]
    doc_ids = [d.doc_id for d in docs_batch]

    # 토큰화
    inputs = tokenizer(
        original_texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )

    # 데이터를 GPU로 로드
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 각 문서당 10개의 쿼리를 생성
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_OUTPUT_LENGTH,
                do_sample=True,
                top_k=10,
                num_return_sequences=NUM_QUERIES
            )

    # 생성된 쿼리 디코딩
    decoded_queries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i, (doc_id, original_text) in enumerate(zip(doc_ids, original_texts)):
        start_idx = i * NUM_QUERIES
        end_idx = start_idx + NUM_QUERIES

        # 문서당 10개의 쿼리를 짤라서 결합
        queries = decoded_queries[start_idx:end_idx]
        queries_text = " ".join(queries)
        expanded_text = f"{original_text} {queries_text}"
        
        results_list.append({
            "doc_id": doc_id,
            "original_text": original_text,
            "generated_queries": queries,
            "text": expanded_text
        })

if __name__ == "__main__":
    main()
