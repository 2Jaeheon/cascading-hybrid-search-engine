import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ir_datasets
from src.core.search_engine import SearchEngine

def main():
    print("=== 인덱싱 프로세스 시작 ===")
    start_time = time.time()
    
    # 서치 엔진 초기화
    engine = SearchEngine(index_path="data/index.pkl")
    
    EXPANDED_DOCS_PATH = "data/expanded_docs.json"
    dataset_id = "wikir/en1k/training"
    documents = []
    titles_map = {}

    # 확장된 문서(JSON)가 있는지 먼저 확인
    if os.path.exists(EXPANDED_DOCS_PATH):
        with open(EXPANDED_DOCS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            doc_id = item['doc_id']
            text = item.get('text', item.get('original_text', ''))
            title = item.get('title', '')

            indexed_text = text
            if title:
                # title을 두 번 넣음
                # 키워드가 title에서 매칭되면 원하는 문서일 가능성이 큼
                indexed_text = f"{title} {title} {text}"
            
            documents.append((doc_id, indexed_text))
            if title:
                titles_map[doc_id] = title
    
    # Doc2Query에서 생성된 JSON문서가 없으면 원본 ir_datasets 사용
    else:
        print("원본 데이터셋 사용")
        dataset = ir_datasets.load(dataset_id)
        
        for doc in dataset.docs_iter():
            documents.append((doc.doc_id, doc.text))
            if len(documents) % 10000 == 0:
                print(f"{len(documents)}개의 문서를 읽었습니다.")
    
    print("인덱스 구축 중...")
    engine.build_index_from_data(documents)
    engine.titles = titles_map
    
    engine.save()
    
    elapsed = time.time() - start_time
    print(f"=== 인덱싱 완료. 소요 시간: {elapsed:.2f}초 ===")

if __name__ == "__main__":
    main()
