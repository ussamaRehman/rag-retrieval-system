import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.app.settings import Settings


def _write_index(tmp_path: Path) -> None:
    chunks = [
        {
            "doc_id": "refund_policy",
            "chunk_id": "refund_policy_0",
            "text": "Refunds are available within 30 days of delivery.",
            "start_offset": 0,
            "end_offset": 55,
        },
        {
            "doc_id": "shipping_policy",
            "chunk_id": "shipping_policy_0",
            "text": "Delivery time is typically 3-5 business days.",
            "start_offset": 0,
            "end_offset": 52,
        },
    ]
    with (tmp_path / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            json.dump(chunk, handle)
            handle.write("\n")
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))
    (tmp_path / "params.json").write_text(
        json.dumps({"embed_model_name": "all-MiniLM-L6-v2"}), encoding="utf-8"
    )


def test_health_and_predict_contract(tmp_path: Path) -> None:
    _write_index(tmp_path)
    settings = Settings(index_dir=str(tmp_path), default_mode="bm25")
    app = create_app(settings)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "versions" in body

    response = client.post("/predict", json={"query": "refund", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["no_answer"] is False
    assert payload["citations"]
    assert payload["citations"][0]["doc_id"] == "refund_policy"


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    settings = Settings(
        index_dir=str(tmp_path),
        rate_limit_rps=0.0,
        rate_limit_burst=0,
        request_timeout_seconds=5.0,
    )
    app = create_app(settings)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 429
    payload = response.json()
    assert payload["error"]["code"] == "rate_limited"


def test_query_too_long_returns_422(tmp_path: Path) -> None:
    _write_index(tmp_path)
    settings = Settings(index_dir=str(tmp_path), max_query_chars=5)
    app = create_app(settings)
    client = TestClient(app)

    response = client.post("/predict", json={"query": "refunds", "top_k": 1})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"


def test_batch_too_large_returns_422(tmp_path: Path) -> None:
    _write_index(tmp_path)
    settings = Settings(index_dir=str(tmp_path), max_batch_size=1)
    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/predict_batch",
        json={"queries": ["refund", "shipping"], "top_k": 1},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"


def test_top_k_too_large_returns_422(tmp_path: Path) -> None:
    _write_index(tmp_path)
    settings = Settings(index_dir=str(tmp_path), max_top_k=1)
    app = create_app(settings)
    client = TestClient(app)

    response = client.post("/predict", json={"query": "refund", "top_k": 5})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"


def test_timeout_returns_504(tmp_path: Path, monkeypatch) -> None:
    _write_index(tmp_path)
    settings = Settings(
        index_dir=str(tmp_path),
        rate_limit_rps=1000.0,
        rate_limit_burst=1000,
        request_timeout_seconds=0.01,
    )
    app = create_app(settings)
    client = TestClient(app)

    def slow_retrieve(*args, **kwargs):
        import time

        time.sleep(0.1)
        return []

    monkeypatch.setattr(app.state.retrieval_service, "retrieve", slow_retrieve)

    response = client.post("/predict", json={"query": "refund", "top_k": 1})
    assert response.status_code == 504
    payload = response.json()
    assert payload["error"]["code"] == "timeout"
