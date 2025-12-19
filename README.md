# 검색엔진 개발 마일스톤

본 프로젝트는 전통적인 키워드 검색(BM25)과 최신 딥러닝 기반의 희소 검색(SPLADE)을 결합한 하이브리드 검색 엔진을 구축하고, 최종적으로 Cross-Encoder를 통해 정밀한 순위 조정을 수행하는 고성능 검색 시스템 개발을 목표로 합니다.

## 1. 프로젝트 개요 및 목표

단순한 키워드 매칭의 한계를 극복하고, 의미 기반 검색의 장점을 결합하여 Recall(재현율)과 Precision(정확도)을 동시에 극대화합니다.
- Core Strategy: Hybrid Retrieval (BM25 + SPLADE)
- Ranking Refinement: Cross-Encoder
- Key Constraint: 외부 검색 엔진(Elasticsearch 등) 없이 Inverted Index 및 핵심 알고리즘 직접 구현

## 2. 전체 시스템 아키텍처

시스템은 속도와 정확도를 모두 잡기 위한 3단계로 동작합니다.

1. Stage1: Retrieval
- BM25 (Lexical): 키워드 매칭
- SPLADE (Semantic): 문맥 및 의미 기반 확장 검색
- Fusion: 두 검색 결과를 결합하여 상위 100개의 후보군을 추출

2. Stage2 : ReRanking
- cross-encoder: 단계에서 선별된 소수의 후보 문서에 대해 질의와 문서를 심층 비교하여 최종 순위를 매깁니다

## 3. 단계별 상세 가이드

### Phase 1. 데이터 준비, 확장 및 인덱싱 (Data Preparation & Expansion)
**목표**: 원본 문서에 잠재된 질의를 추가하여(Expansion) 검색 확률을 높이고, 이를 Inverted Index로 구축합니다.

#### 1. 데이터 로드 및 문서 확장 (Doc2Query)
- **데이터**: `ir_datasets`의 `wikir/en1k/training`
- **Doc2Query**:
  - `scripts/expand_docs.py`: `castorini/doc2query-t5-base-msmarco` 모델을 사용합니다.
  - 각 문서마다 10개의 예상 질의를 생성하여 문서 뒤에 덧붙입니다.
  - **결과물**: `data/expanded_docs.json`
- 실제 인덱싱은 확장된 문서를 기반으로 진행합니다.

#### 2. 전처리 (Preprocessing)
- **Tokenization**: `bert-base-uncased` 모델의 BERT Tokenizer를 사용합니다.
  - 단순 공백/규칙 기반이 아닌, 의미 단위의 Subword Tokenization을 수행합니다.

#### 3. Inverted Index 구축
- `data/expanded_docs.json`의 확장된 텍스트를 입력으로 받습니다.
- BM25 통계: 확장된 텍스트 길이를 기준으로 `avgdl`, `doc_len`을 계산하여 저장합니다.

### Phase 2. 검색 (BM25 + SPLADE)

**목표**: 키워드 매칭(BM25)과 문맥 매칭(SPLADE)을 결합하여 Recall을 극대화합니다.

#### 1. BM25
- 전통적인 키워드 매칭 방식.
- **알고리즘**: TF-IDF의 개선판으로, 문서 길이와 단어 빈도를 고려하여 점수를 매깁니다.
- **역할**: 사용자가 입력한 단어가 정확히 포함된 문서를 놓치지 않고 찾아냅니다.

#### 2. SPLADE
- 문맥을 고려하여 문서에 단어가 없어도 의미가 통하는 문서를 찾아냅니다.
- **역할**: "Car"를 검색했을 때 "Vehicle"이나 "Automobile"이 포함된 문서를 찾아냅니다.

#### 3. Ensemble
- **방식**: Reciprocal Rank Fusion (RRF) 알고리즘을 사용하여 결합합니다.
  - `Score = 1 / (k + Rank_BM25) + 1 / (k + Rank_SPLADE)` (통상 `k=60`)
- **효과**: scale이 서로 다른 두 모델을 Rank 기반으로 공정하게 합쳐, 한쪽 모델의 이상치 왜곡을 방지합니다.
- **결과**: 안정적으로 상위 50개의 후보군을 추출합니다.

### Phase 3. 정밀 리랭킹

**목표**: 하이브리드 검색으로 추려진 50개의 문서를 "진짜 정답" 순서대로 재배열합니다. (Precision 확보)

#### Cross-Encoder 로직
- **모델**: `cross-encoder/ms-marco-MiniLM-L-6-v2` 등 성능과 속도가 균형 잡힌 모델 사용.
- **입력**: Query와 문서를 `[SEP]` 토큰으로 이어 붙여 모델에 동시에 넣습니다.
- **계산**: 모델이 query와 문서 사이의 모든 토큰 관계를 분석하여 관련성 점수를 출력합니다.
- **결과**: 가장 관련성이 높은 상위 10개 문서를 최종 검색 결과로 확정합니다.

### Phase 4. UI 및 성능 평가

**목표**: 구축한 검색 엔진의 성능을 수치로 증명하고, 웹 인터페이스로 시각화합니다.

#### 1. 성능 평가 (TREC Eval)
- `pytrec_eval`을 사용하여 표준 지표를 측정합니다.
- **MAP**: 전체적인 랭킹 품질
- **nDCG@10**: 상위 10개 결과의 가중치 반영 품질
- **P@10**: 상위 10개 중 정답 비율



#### 2. Web UI (FastAPI)
- **Frontend**: Jinja2 Template (HTML/CSS)
- **Backend**: FastAPI
- **기능**: 검색어 입력 -> 하이브리드 검색 -> 리랭킹 -> 결과 출력 (점수 및 소요시간 포함)

## 4. 프로젝트 진행 체크리스트

### 1주차: 기본 인덱싱 및 BM25
- [X] 데이터셋(`wikir/en1k`) 다운로드 및 내용 확인
- [X] 전처리 모듈 및 Positional Inverted Index 구현
- [X] **BM25 알고리즘** 구현 및 테스트

### 2주차: SPLADE 및 하이브리드 구현
- [ ] **SPLADE 모델** 연동 및 문서 인코딩
- [ ] **Hybrid Search** (BM25 + SPLADE) 로직 구현

### 3주차: 리랭킹 및 시스템 통합
- [ ] **Cross-Encoder** 활용 Reranker 구현
- [ ] 전체 파이프라인 통합 및 테스트

### 4주차: 평가 및 UI
- [ ] **TREC Eval** 성능 평가
- [ ] **Web UI** 개발 및 보고서 작성

---

## 5. 핵심 요약 테이블

| 단계 | 기법 (Model) | 역할 | 특징 | 대상 문서 수 |
|------|------|------|------|:---:|
| 1 | BM25 (Lexical) | 키워드 매칭 | 정확한 단어 매칭에 강함 | 전체 (N) |
| 1' | SPLADE (Semantic) | 문맥 매칭 | 의미적 유사성 파악 | 전체 (N) |
| 2 | Hybrid (Ensemble) | 후보군 추출 | 위 두 점수를 결합 (Recall↑) | Top 50 추출 |
| 3 | Cross-Encoder | 정밀 재정렬 | 문장 간 심층 관계 파악 (Precision↑) | 50 → Top 10 |

---
