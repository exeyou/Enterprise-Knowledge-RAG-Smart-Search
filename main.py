import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.schemas import IngestResponse, QueryRequest, QueryResponse
from backend.core.ingestion import DocumentIngestor
from backend.core.indexer import KnowledgeIndexer
from backend.core.retriever import HybridRetriever
from backend.core.router import ModelRouter
from backend.core.generator import GroundedGenerator

app = FastAPI(title="Enterprise Knowledge RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestor = DocumentIngestor(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
indexer = KnowledgeIndexer()
router = ModelRouter()
generator = GroundedGenerator()

processed_nodes_registry = []


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    global processed_nodes_registry
    saved_files = []
    total_chunks = 0

    for file in files:
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            nodes = ingestor.process_file(file_path)
            if nodes:
                indexer.add_nodes(nodes)
                processed_nodes_registry.extend(nodes)
                total_chunks += len(nodes)
                saved_files.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error processing file {file.filename}: {str(e)}")

    return IngestResponse(
        status="success", processed_files=saved_files, total_chunks_added=total_chunks
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    global processed_nodes_registry

    if not processed_nodes_registry:
        processed_nodes_registry = []

    selected_model, route_reason = router.route_query(
        query=request.query, force_flagship=request.force_flagship
    )

    retriever = HybridRetriever(
        vector_store=indexer.vector_store, all_nodes=processed_nodes_registry
    )
    relevant_nodes = retriever.retrieve(request.query)

    response = generator.generate_answer(
        query=request.query,
        context_nodes=relevant_nodes,
        model_name=selected_model,
        route_reason=route_reason,
    )

    return response