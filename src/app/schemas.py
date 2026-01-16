from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RetrievalMode = Literal["bm25", "dense"]


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    snippet: str
    start_offset: int
    end_offset: int


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, gt=0)
    mode: Optional[RetrievalMode] = None


class PredictBatchRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    top_k: int = Field(5, gt=0)
    mode: Optional[RetrievalMode] = None


class PredictResponse(BaseModel):
    answer: str
    no_answer: bool
    citations: List[Citation]
    versions: Dict[str, str]
    request_id: str


class PredictBatchResponse(BaseModel):
    responses: List[PredictResponse]
    versions: Dict[str, str]
    request_id: str


class HealthResponse(BaseModel):
    status: str
    versions: Dict[str, str]


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, str]] = None


class ErrorResponse(BaseModel):
    error: ErrorInfo
    request_id: str
