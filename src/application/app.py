from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.search_engine import SearchEngine
import contextlib
import ir_datasets
import time
import os
import re

# 전역 인스턴스
engine: SearchEngine = None
DOC_STORE = {} # {doc_id: text}

# 현재 파일의 디렉토리 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 불용어(stopwords) 목록 - 하이라이트에서 제외
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'want', 'would', 'could'
}


def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
        
    terms = query.lower().split()
    meaningful_terms = [t for t in terms if t not in STOPWORDS and len(t) > 2]
    
    if not meaningful_terms:
        return text
    
    for term in meaningful_terms:
        escaped_term = re.escape(term)
        pattern = re.compile(rf'\b({escaped_term})\b', re.IGNORECASE)
        text = pattern.sub(r"<mark>\1</mark>", text)
        
    return text

# 수명 주기 관리를 위한 함수
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # init(초기화)
    global engine
    
    print("엔진 초기화중...")
    engine = SearchEngine(index_path="data/index.pkl")
    
    if not engine.load():
        print("인덱스 로드 실패. 'scripts/run_indexing.py'를 먼저 실행해주세요.")
    else:
        print("인덱스 로드 성공.")

    print("문서를 메모리에 로드중...")
    start_time = time.time()
    dataset = ir_datasets.load("wikir/en1k/training")

    for doc in dataset.docs_iter():
        DOC_STORE[doc.doc_id] = doc.text
    print(f"문서 로드 완료: {len(DOC_STORE)}, {time.time() - start_time:.2f}초")
    
    print("SPLADE 모델 로딩 중...")
    engine.load_splade_model()
    engine.hybrid_search("warm up!!", top_k=100)
    print("모델 로딩 완료.")

    yield

    # 종료
    engine = None
    DOC_STORE.clear()

app = FastAPI(lifespan=lifespan)

# 정적 파일 및 템플릿 경로 설정
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", page: int = 1):
    results = []
    search_time = 0.0
    limit = 10
    
    if q and engine:
        start_time = time.time()
        offset = (page - 1) * limit

        results_with_scores = engine.hybrid_search(q, top_k=limit, offset=offset)
        
        for rank, (doc_id, score) in enumerate(results_with_scores, offset + 1):
            text = DOC_STORE.get(doc_id, "Content not found.")
            title = engine.titles.get(doc_id, "제목 없음")
            snippet = text[:300] + "..." if len(text) > 300 else text
            snippet = highlight_text(snippet, q)
            
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "snippet": snippet,
                "full_text": text,
                "score": f"{score:.4f}"
            })
            
        search_time = time.time() - start_time
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request, 
            "query": q, 
            "results": results, 
            "search_time": f"{search_time:.4f}",
            "page": page,
            "has_next": len(results) == limit
        }
    )