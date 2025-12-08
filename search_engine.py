import ir_datasets
from typing import List, Tuple, Dict
import os
import pickle

class SearchEngine:
    def __init__(self, dataset_id: str = "wikir/en1k/training"):
        """
        검색 엔진 객체 초기화
        """
        self.dataset_id = dataset_id
        
        # 데이터셋 로드
        self.dataset = ir_datasets.load(dataset_id)
        
        # 역색인 (inveted Index)
        self.index: Dict[str, List[Tuple[str, int]]] = {}
        
        # 문서 길이 정보
        self.doc_len: Dict[str, int] = {}
        
        # 문서 빈도
        self.df: Dict[str, int] = {}
        
        # 전체 통계 정보
        self.doc_count = 0 # 전체 문서 수
        self.avg_doc_len = 0.0 # 평균 문서 길이

    def build_index(self):
        """
        문서를 읽으며 InvertedIndex를 구축
        """
        print(f"{self.dataset_id}에 대한 인덱스를 구축합니다.")        
        # TODO: 토큰화 및 카운팅 구현 필요
        pass

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        입력된 쿼리에 대해 BM25 점수를 계산하고, 상위 k개 문서를 반환하는 함수.
        실제 프론트에서 이 함수를 호출해서 결과를 보여주는 것!!!
        """
        print(f"{query}에 대해서 검색하였습니다.")
        # TODO: 계산하는 로직 구현이 필요
        return [("dummy1", 1.5), ("dummy2", 0.9)]

    def save_index(self, path: str):
        """
        메모리에 있는 인덱스 데이터를 파일로 저장
        """
        print(f"인덱스를 다음 경로({path})에 저장합니다.")
        # TODO: 데이터 저장
        pass

    def load_index(self, path: str) -> bool:
        """
        파일에서 인덱스 데이터를 불러와 메모리에 적재
        """
        if not os.path.exists(path):
            return False
            
        print(f"인덱스를 다음 경로({path})에서 로드합니다.")
        # TODO: 데이터 복원
        return True
