from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from src.app.errors import add_exception_handlers
from src.app.retrieval_service import RetrievalService
from src.app.schemas import HealthResponse, PredictBatchRequest, PredictRequest, PredictResponse
from src.app.settings import Settings

PREDICT_REQUESTS = Counter(
    "rag_predict_requests_total",
    "Total predict requests.",
    ["endpoint", "mode"],
)


def _build_answer(citations: list[dict]) -> tuple[str, bool]:
    if not citations:
        return "I don't know based on the provided documents.", True
    snippets = [citation["snippet"] for citation in citations[:2]]
    answer = " ".join(snippets).strip()
    return answer, False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service = RetrievalService(
        Path(settings.index_dir),
        api_version=settings.api_version,
        max_top_k=settings.max_top_k,
        snippet_chars=settings.snippet_chars,
        default_mode=settings.default_mode,
    )

    app = FastAPI(title="RAG Retrieval API", version=settings.api_version)
    app.state.retrieval_service = service

    add_exception_handlers(app)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", versions=service.versions())

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        citations = service.retrieve(payload.query, payload.mode, payload.top_k)
        answer, no_answer = _build_answer(citations)
        mode = payload.mode or service.default_mode
        PREDICT_REQUESTS.labels(endpoint="/predict", mode=mode).inc()
        return PredictResponse(
            answer=answer,
            no_answer=no_answer,
            citations=citations,
            versions=service.versions(),
            request_id=request.state.request_id,
        )

    @app.post("/predict_batch", response_model=list[PredictResponse])
    def predict_batch(
        payload: PredictBatchRequest, request: Request
    ) -> list[PredictResponse]:
        queries = payload.queries
        top_k = payload.top_k
        mode = payload.mode
        responses = []
        for query in queries:
            citations = service.retrieve(query, mode, top_k)
            answer, no_answer = _build_answer(citations)
            responses.append(
                PredictResponse(
                    answer=answer,
                    no_answer=no_answer,
                    citations=citations,
                    versions=service.versions(),
                    request_id=request.state.request_id,
                )
            )
        PREDICT_REQUESTS.labels(
            endpoint="/predict_batch", mode=mode or service.default_mode
        ).inc(len(queries))
        return responses

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
