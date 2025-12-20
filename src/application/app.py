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


def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
        
    # 검색어를 단어 단위로 분리
    terms = query.split()
    
    # 각 단어에 대해 대소문자 무시하고 치환
    for term in terms:
        escaped_term = re.escape(term)
        pattern = re.compile(f"({escaped_term})", re.IGNORECASE)
        text = pattern.sub(r"<mark>\1</mark>", text)
        
    return text

# 수명 주기 관리를 위한 함수
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 초기화
    global engine
    
    print("엔진 초기화중...")
    # index_path는 프로젝트 루트 기준 data/index.pkl
    engine = SearchEngine(index_path="data/index.pkl")
    
    # 인덱스가 존재하는지 확인하고 로드
    if not engine.load():
        print("인덱스 로드 실패. 'scripts/run_indexing.py'를 먼저 실행해주세요.")
    else:
        print("인덱스 로드 성공.")

    # 문서 내용 메모리에 로드
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
async def search(request: Request, q: str = ""):
    results = []
    search_time = 0.0
    
    if q and engine:
        start_time = time.time()
        results_with_scores = engine.hybrid_search(q, top_k=10)
        
        for rank, (doc_id, score) in enumerate(results_with_scores, 1):
            text = DOC_STORE.get(doc_id, "Content not found.")
            
            title = engine.titles.get(doc_id, "제목 없음")
            
            snippet = text[:300] + "..." if len(text) > 300 else text
            
            snippet = highlight_text(snippet, q)
            
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "snippet": snippet,
                "score": f"{score:.4f}"
            })
            
        search_time = time.time() - start_time
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": q, "results": results, "search_time": f"{search_time:.4f}"}
    )