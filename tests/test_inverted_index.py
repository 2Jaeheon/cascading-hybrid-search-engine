import pytest
import os
from src.index.inverted_index import InvertedIndex

class TestInvertedIndex:
    @pytest.fixture
    def index_engine(self):
        return InvertedIndex()

    def test_add_document_position(self, index_engine):
        # Positional Index 구조 검증
        # Given
        doc_id = "doc_1"
        text = "apple banana apple"  # apple이 0번째, 2번째에 등장
        
        # When
        index_engine.add_document(doc_id, text)
        
        # Then
        apple_term = index_engine.tokenizer.tokenize("apple")[0]
        
        assert apple_term in index_engine.index
        assert doc_id in index_engine.index[apple_term]
        
        positions = index_engine.index[apple_term][doc_id]
        assert positions == [0, 2]

    def test_statistics_calculation(self, index_engine):
        # 문서 통계(평균 길이, 문서 수) 계산 검증
        # Given
        index_engine = InvertedIndex() # fixture 대신 새로 생성하여 깨끗한 상태 보장
        
        # When
        index_engine.add_document("doc1", "python java")
        index_engine.add_document("doc2", "python")
        index_engine.finalize()
        
        # Then
        assert index_engine.doc_count == 2
        assert index_engine.doc_lengths["doc1"] == 2
        assert index_engine.doc_lengths["doc2"] == 1
        assert index_engine.avg_doc_len == 1.5

    def test_save_and_load(self, index_engine, tmp_path):
        # 인덱스 저장 및 로드 테스트
        # Given
        index_engine.add_document("test_doc", "search engine test")
        index_engine.finalize()
        save_file = tmp_path / "test_index.pkl"
        
        # When
        index_engine.save(str(save_file))
        assert os.path.exists(save_file)
        
        new_index = InvertedIndex()
        success = new_index.load(str(save_file))
        
        # Then
        assert success is True
        assert new_index.doc_count == 1
        
        term = index_engine.tokenizer.tokenize("engine")[0]
        assert term in new_index.index
