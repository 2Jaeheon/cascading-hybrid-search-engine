from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from search_engine import SearchEngine
import contextlib

# 전역 인스턴스
engine: SearchEngine = None

# 수명 주기 관리를 위한 함수
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 초기화
    global engine
    engine = SearchEngine()

    # 인덱스 로딩 딱 한 번
    
    yield

    # 종료
    engine = None

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    results = []
    if q and engine:
        results = engine.search(q, top_k=10)
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": q, "results": results}
    )