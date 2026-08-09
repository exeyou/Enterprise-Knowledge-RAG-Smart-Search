from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str
    processed_files: List[str]
    total_chunks_added: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="Target query text for knowledge lookup")
    top_k: int = Field(default=5, description="Number of context chunks to retrieve")
    force_flagship: bool = Field(
        default=False, description="Flag to force high-tier LLM execution"
    )


class SourceCitation(BaseModel):
    document_name: str
    page_label: Optional[str] = "N/A"
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]
    model_used: str
    route_reasoning: str