import sys
import os
import json
import time
from tqdm import tqdm
from typing import List, Tuple
import ir_datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.splade_model import SpladeModel
from src.core.splade_index import SpladeIndex

def main():
    print("=== SPLADE 인덱싱 프로세스 시작 ===")
    start_time = time.time()

    DATA_PATH = "data/expanded_docs.json"
    INDEX_PATH = "data/splade_index"
    BATCH_SIZE = 32
    DATASET_ID = "wikir/en1k/training"
    
    model = SpladeModel() 
    index = SpladeIndex()
    documents: List[Tuple[str, str]] = []

    # 데이터 로딩 및 처리
    if os.path.exists(DATA_PATH):
        print(f"확장된 데이터셋 로드 중: {DATA_PATH}")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    doc_id = item.get('id', str(item.get('doc_id')))
                    text = item.get('text', item.get('original_text', ''))
                    title = item.get('title', '')
                    
                    if title:
                        text = f"{title} {text}"
                        
                    documents.append((doc_id, text))
            
            elif isinstance(data, dict):
                for doc_id, text in data.items():
                    documents.append((doc_id, text))
                    
    else:
        print(f"확장된 문서가 없습니다. 원본 데이터셋({DATASET_ID})을 사용합니다.")
        try:
            dataset = ir_datasets.load(DATASET_ID)
            for doc in dataset.docs_iter():
                documents.append((doc.doc_id, doc.text))
                if len(documents) % 10000 == 0:
                    print(f"{len(documents)}개의 문서를 읽었습니다.")
        except Exception as e:
            print(f"데이터셋 로드 실패: {e}")
            return

    # 배치 처리 루프
    total_docs = len(documents)
    print(f"인덱싱 시작 (총 {total_docs}개 문서, 배치 크기: {BATCH_SIZE})")
    
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Indexing"):
        batch_docs = documents[i:i + BATCH_SIZE]
        
        batch_ids = [doc[0] for doc in batch_docs]
        batch_texts = [doc[1] for doc in batch_docs]
        
        try:
            sparse_vectors = model.encode_batch(batch_texts, batch_size=BATCH_SIZE)
            
            index.add_batch(
                batch_ids, 
                sparse_vectors['indices'], 
                sparse_vectors['values']
            )
        except Exception as e:
            print(f"배치 처리 중 오류 발생 (Index {i}): {e}")
            continue

    # 빌드 및 저장
    index.build()
    index.save(INDEX_PATH)
    
    elapsed = time.time() - start_time
    print(f"=== SPLADE 인덱싱 완료. 소요 시간: {elapsed:.2f}초 ===")

if __name__ == "__main__":
    main()
