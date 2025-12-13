# 검색엔진 개발 마일스톤

## 1. 프로젝트 개요 및 목표

이 프로젝트의 핵심은 검색엔진(BM25)를 직접 구축하여 AI Model을 접목하여 성능을 극대화하는 것입니다.
- **필수 요건**: 외부 검색 엔진(Elasticsearch 등) 사용 금지, Inverted Index 및 BM25 알고리즘 직접 구현
- **파이프라인**: `BM25 (Base)` -> `Positional Index (Phrase)` -> `PRF (Expansion)` -> `SPLADE (Coarse Reranking)` -> `Cross-Encoder (Fine Reranking)`

## 2. 전체 시스템 아키텍처

시스템은 속도와 정확도를 모두 잡기 위한 3단계로 동작합니다.

1. **전처리 및 인덱싱**: 문서를 분석해 Inverted Index와 Vector Index를 미리 구축합니다.
2. **Level 1**: **BM25** 알고리즘으로 대규모 문서 중 후보군 **1,000개**를 빠르게 확보합니다. (Recall)
3. **Query Expansion**: 1차 결과에서 힌트를 얻어 질의를 확장(PRF)합니다.
4. **Level 2 (Coarse Reranking)**: **SPLADE**로 문맥을 고려해 상위 **50개**를 추려냅니다. (Semantic Filter)
5. **Level 3 (Fine Reranking)**: **Cross-Encoder**로 최종 50개를 정밀 채점하여 순위를 갱신합니다. (Precision)

## 3. 단계별 상세 가이드

### Phase 1. 데이터 준비 및 인덱싱
**목표**: `wikir/en1k` 데이터셋을 읽어 검색 가능한 형태(invertedIndex)로 변환합니다.

#### 1. 데이터 로드
- `ir_datasets` 라이브러리를 사용해 `wikir/en1k/training` 데이터를 불러옵니다.
- 문서(Document), 질의(Query), 정답지(Qrels) 3가지를 테스트합니다.

#### 2. 전처리 (Preprocessing)
- **토큰화(Tokenization)**: 문장을 단어 단위로 쪼갭니다. (영어이므로 공백 기준 + 소문자 변환 + 특수문자 제거)
- **불용어(stopword)제거**: `nltk` 라이브러리의 확장된 불용어 리스트를 사용하여 `the`, `is`, `a` 등 의미 없는 단어를 제거합니다.
- **어간 추출(Stemming)**: `PorterStemmer`를 적용하여 `running`, `runs` -> `run`과 같이 원형으로 통일합니다.

#### 3. Positional Inverted Index 구축 (가산점 핵심)
단순히 "이 단어가 있다/없다"만 저장하지 말고, 몇번째 위치에 있는지 저장합니다.

- **구조**: `Dictionary<단어, Dictionary<문서ID, List<위치>>>`
- **예시**: "AI"라는 단어가 문서 5번의 3번째, 10번째에 등장한다면?
  - `"ai": { 5: [3, 10], ... }`
- **통계 저장**: BM25 계산을 위해 아래 정보도 미리 계산해 저장해 둡니다.
  - `N`: 전체 문서 개수
  - `avgdl`: 평균 문서 길이
  - `doc_len`: 각 문서의 길이

### Phase 2. 1차 검색: BM25 & 구 검색 (Retrieval)

**목표**: 사용자의 질의와 관련 있을 법한 후보 문서 1,000개를 최대한 빠르게 찾습니다. (Recall 확보)

#### 1. BM25 알고리즘 구현
- **TF (Term Frequency)**: 단어가 문서에 얼마나 자주 나오는가? (많을수록 좋음)
- **IDF (Inverse Document Frequency)**: 단어가 얼마나 희귀한가? (희귀할수록 점수 높음)
- **Length Normalization**: 문서가 너무 길지 않은가? (짧은 문서에 매칭될수록 가산점)
- **구현**: 위 3가지 요소를 결합한 BM25 수식을 코드로 옮깁니다. (파라미터 `k1=1.5, b=0.75` 권장)

#### 2. 구(Phrase) 검색 로직 추가 (Positional Index 활용)
- 질의에 연속된 단어(예: "Solar Energy")가 있다면, 인덱스에서 두 단어의 위치 정보를 확인합니다.
- 실제 문서에서도 두 단어가 붙어있다면(위치 차이가 1이라면) BM25 점수에 가산점을 줍니다.
- **효과**: "New York"을 검색했을 때 "New"와 "York"가 따로 떨어진 문서보다, 붙어있는 문서를 상위에 올립니다.

### Phase 3. 질의 확장: Pseudo-Relevance Feedback (Expansion)

**목표**: 사용자의 짧고 모호한 검색어를 구체화하여 숨겨진 의도를 파악합니다.

#### 작동 프로세스
1. **가정**: "BM25로 찾은 상위 10개 문서는 정답일 확률이 높다."
2. **분석**: 상위 10개 문서에서 공통적으로 자주 등장하는 핵심 키워드 3~5개를 뽑습니다.
3. **확장**: 원래 질의 뒤에 뽑힌 키워드를 덧붙입니다.
   - User Query: "Apple"
   - Top Docs Topic: "iPhone", "Mac", "Technology"
   - Expanded Query: "Apple *iPhone Mac Technology*"
4. **효과**: 사용자가 "iPhone"을 입력하지 않았어도, 관련된 문서를 찾아낼 수 있게 됩니다.

### Phase 4. 고도화된 리랭킹 (Cascading Rerank)

**목표**: 속도와 정확도의 균형을 맞추기 위해 3단계 파이프라인으로 구성하였습니다.

#### 1. Level 1: 후보군 정제
- **입력**: BM25 & PRF를 거친 상위 1,000개 문서
- **모델**: **SPLADE (Bi-Encoder)**
- **로직**:
  - 문맥을 고려하는 SPLADE 모델로 문서와 질의를 Sparse Vector로 변환합니다.
  - 내적 계산을 통해 1,000개 문서와의 유사도를 계산합니다.
  - 상위 **50개**의 고품질 문서를 선별합니다.

#### 2. Level 2: 재정렬
- **입력**: 선별된 상위 50개 문서
- **모델**: **Cross-Encoder** (예: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **로직**:
  - 질의와 문서를 `[SEP]` 토큰으로 이어 붙여 모델에 "통째로" 입력합니다. (`Model(Query, Doc)`)
  - 질의와 문서의 토큰간의 attention을 계산하여 의미적 적합성을 정밀 채점합니다.
  - 최종 점수 순으로 재정렬하여 TOP 10을 확정합니다.

### Phase 5. UI 및 성능 평가

**목표**: 시스템이 얼마나 잘 작동하는지 보여주고 수치로 증명합니다.

#### 1. 검색 UI 구현
- **기술 스택**: **FastAPI** (Backend) + **Jinja2 Templates** (Frontend)
- **필수 포함 정보**:
  - 검색 소요 시간
  - 검색된 총 문서 수
  - 결과 리스트 (순위, 문서 제목, 문서 내용 일부, 관련도 점수)

#### 2. 성능 평가 (TREC Eval)
- **도구**: `pytrec_eval` 라이브러리
- **지표**:
  - **MAP (Mean Average Precision)**: 전반적인 검색 품질
  - **nDCG**: 순위의 정확도 (상위에 정답이 있는가?)
  - **P@10 (Precision at 10)**: 1페이지 내에 정답이 몇 개 있는가?

## 4. 프로젝트 진행 체크리스트

### 1주차: 기본 구현
- [X] 데이터셋(`wikir/en1k`) 다운로드 및 내용 확인 (Doc, Query, Qrels)
- [X] 전처리 모듈 구현 (토큰화, 불용어 제거)
- [X] **Positional Inverted Index** 구현 (직접 구현 필수)
- [X] 인덱스 구축 후 저장/로드 기능 확인

### 2주차: 검색 엔진 코어 개발
- [ ] **BM25 알고리즘** 구현 및 1차 검색 테스트
- [ ] `TREC Eval`로 BM25 베이스라인 성능 측정 (기록해둘 것)
- [ ] **Pseudo-Relevance Feedback (PRF)** 모듈 구현 및 질의 확장 테스트

### 3주차: 고도화 및 마무리
- [ ] **SPLADE + FAISS** 연동 (Coarse Reranking)
- [ ] **Cross-Encoder** 연동 (Fine Reranking)
- [ ] 최종 파이프라인 연결 및 성능 측정
- [ ] 간단한 검색 UI 개발
- [ ] **보고서 작성** (Multi-stage 아키텍처 다이어그램 포함)

---

## 5. 핵심 요약 테이블

| 단계 | 기법 (Model) | 역할 | 특징 | 대상 문서 수 |
|------|------|------|------|:---:|
| 0 | Inverted Index | 빠른 탐색 | 키워드 매칭 | 전체 (N) |
| 1 | BM25 + PRF | Recall 확보 | 통계적 확률 기반 | 1,000개 추출 |
| 2 | SPLADE (Reranking) | Meaning Filter | 문맥 파악 & 고속 벡터 연산 | 1,000 → 50 |
| 3 | Cross-Encoder | Precision 확보 | 문장 쌍 정밀 비교 | 50 → 10 |

---

