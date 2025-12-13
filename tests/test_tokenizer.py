import pytest
from src.utils.tokenizer import Tokenizer

class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return Tokenizer(use_stopwords=True)

    def test_basic_tokenization(self, tokenizer):
        # 소문자, 공백, 특수문자 제거 테스트
        # Given
        text = "Hello, World!"

        # When
        tokens = tokenizer.tokenize(text)

        # Then
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in "".join(tokens)

    def test_stopword_removal(self, tokenizer):
        # stopword 제거 테스트
        # Given
        text = "This is a book about the python"
        
        # When
        tokens = tokenizer.tokenize(text)
        
        # Then
        assert "is" not in tokens
        assert "the" not in tokens
        assert "book" in tokens
        assert "python" in tokens

    def test_stemming(self, tokenizer):
        # stemming 테스트
        # Given
        text_running = "running runs"
        text_compute = "computation computer"

        # When
        tokens_running = tokenizer.tokenize(text_running)
        tokens_compute = tokenizer.tokenize(text_compute)

        # Then
        assert tokens_running == ["run", "run"]
        assert tokens_compute[0] == tokens_compute[1]

    def test_empty_string(self, tokenizer):
        # 빈 문자열 입력 시 빈 리스트 반환 테스트
        # Given
        empty_text = ""
        none_text = None

        # When
        result_empty = tokenizer.tokenize(empty_text)
        result_none = tokenizer.tokenize(none_text)

        # Then
        assert result_empty == []
        assert result_none == []
