from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from src.app.errors import add_exception_handlers, error_response
from src.app.middleware import add_middlewares
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


def _validation_error(message: str, request_id: str):
    return error_response(
        code="validation_error",
        message=message,
        request_id=request_id,
        status_code=422,
    )


async def _retrieve_with_timeout(service: RetrievalService, query: str, mode: str | None, top_k: int, timeout: float):
    try:
        if timeout is not None and timeout >= 0:
            return await asyncio.wait_for(
                asyncio.to_thread(service.retrieve, query, mode, top_k),
                timeout=timeout,
            )
        return await asyncio.to_thread(service.retrieve, query, mode, top_k)
    except (asyncio.TimeoutError, TimeoutError):
        raise TimeoutError from None


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
    app.state.settings = settings

    add_exception_handlers(app)
    add_middlewares(app, settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", versions=service.versions())

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest, request: Request):
        request_id = request.state.request_id
        if len(payload.query) > settings.max_query_chars:
            return _validation_error(
                f"query exceeds {settings.max_query_chars} characters",
                request_id,
            )
        if payload.top_k > settings.max_top_k:
            return _validation_error(
                f"top_k exceeds {settings.max_top_k}",
                request_id,
            )
        try:
            citations = await _retrieve_with_timeout(
                service,
                payload.query,
                payload.mode,
                payload.top_k,
                settings.request_timeout_seconds,
            )
        except TimeoutError:
            return error_response(
                code="timeout",
                message="Request timed out.",
                request_id=request_id,
                status_code=504,
            )
        answer, no_answer = _build_answer(citations)
        mode = payload.mode or service.default_mode
        PREDICT_REQUESTS.labels(endpoint="/predict", mode=mode).inc()
        return PredictResponse(
            answer=answer,
            no_answer=no_answer,
            citations=citations,
            versions=service.versions(),
            request_id=request_id,
        )

    @app.post("/predict_batch", response_model=list[PredictResponse])
    async def predict_batch(
        payload: PredictBatchRequest, request: Request
    ) -> list[PredictResponse]:
        request_id = request.state.request_id
        queries = payload.queries
        top_k = payload.top_k
        mode = payload.mode
        if len(queries) > settings.max_batch_size:
            return _validation_error(
                f"batch size exceeds {settings.max_batch_size}",
                request_id,
            )
        if top_k > settings.max_top_k:
            return _validation_error(
                f"top_k exceeds {settings.max_top_k}",
                request_id,
            )
        for query in queries:
            if len(query) > settings.max_query_chars:
                return _validation_error(
                    f"query exceeds {settings.max_query_chars} characters",
                    request_id,
                )
        responses = []
        for query in queries:
            try:
                citations = await _retrieve_with_timeout(
                    service,
                    query,
                    mode,
                    top_k,
                    settings.request_timeout_seconds,
                )
            except TimeoutError:
                return error_response(
                    code="timeout",
                    message="Request timed out.",
                    request_id=request_id,
                    status_code=504,
                )
            answer, no_answer = _build_answer(citations)
            responses.append(
                PredictResponse(
                    answer=answer,
                    no_answer=no_answer,
                    citations=citations,
                    versions=service.versions(),
                    request_id=request_id,
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
