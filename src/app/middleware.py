from __future__ import annotations

import json
import time
from typing import Dict, Optional
from uuid import uuid4

import anyio
from fastapi import Request
from fastapi.responses import Response

from src.app.errors import error_response
from src.app.settings import Settings


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = max(rate, 0.0)
        self.capacity = max(burst, 0)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated)
        self.updated = now
        if self.rate > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimiter:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = anyio.Lock()

    async def allow(self, key: str) -> bool:
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(self.rate, self.burst)
                self._buckets[key] = bucket
            return bucket.consume()


def add_middlewares(app, settings: Settings) -> None:
    limiter = RateLimiter(settings.rate_limit_rps, settings.rate_limit_burst)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id

        size_response = await _enforce_request_limits(request, settings, request_id)
        if size_response is not None:
            size_response.headers["X-Request-Id"] = request_id
            return size_response

        client_ip = request.client.host if request.client else "unknown"
        if settings.rate_limit_rps >= 0 and settings.rate_limit_burst >= 0:
            allowed = await limiter.allow(client_ip)
            if not allowed:
                response = error_response(
                    code="rate_limited",
                    message="Rate limit exceeded.",
                    request_id=request_id,
                    status_code=429,
                )
                response.headers["X-Request-Id"] = request_id
                return response

        try:
            if settings.timeout_seconds is not None and settings.timeout_seconds >= 0:
                with anyio.fail_after(settings.timeout_seconds):
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except TimeoutError:
            response = error_response(
                code="timeout",
                message="Request timed out.",
                request_id=request_id,
                status_code=504,
            )
        response.headers["X-Request-Id"] = request_id
        return response


async def _enforce_request_limits(
    request: Request, settings: Settings, request_id: str
) -> Optional[Response]:
    if settings.max_request_bytes <= 0:
        return None
    if request.method not in {"POST", "PUT", "PATCH"}:
        return None

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_request_bytes:
                return error_response(
                    code="payload_too_large",
                    message="Request body too large.",
                    request_id=request_id,
                    status_code=413,
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > settings.max_request_bytes:
        return error_response(
            code="payload_too_large",
            message="Request body too large.",
            request_id=request_id,
            status_code=413,
        )

    if request.url.path == "/predict_batch" and body:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        queries = payload.get("queries")
        if isinstance(queries, list) and len(queries) > settings.max_batch:
            return error_response(
                code="batch_too_large",
                message=f"Batch size exceeds {settings.max_batch}.",
                request_id=request_id,
                status_code=413,
            )
    return None
