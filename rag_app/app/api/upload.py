"""
POST /upload

Pipeline: PDF -> extract text -> clean -> chunk -> embed -> store in Postgres.
"""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Document, DocumentChunk
from app.schemas.upload import UploadResponse
from app.services import embeddings
from app.services.chunker import chunk_text
from app.services.document_loader import DocumentLoadError, extract_text_from_pdf
from app.services.embeddings import EmbeddingError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in this version.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # 1. Extract + clean text
    try:
        text = extract_text_from_pdf(file_bytes)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content could be chunked from this document.")

    # 3. Embed
    try:
        vectors = embeddings.embed_texts(chunks)
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 4. Store document + chunks in Postgres
    try:
        document = Document(filename=file.filename, content=text)
        db.add(document)
        db.flush()  # get document.id before committing

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_text=chunk,
                    embedding=vector,
                    chunk_index=idx,
                )
            )
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.error("Failed to store document/chunks: %s", exc)
        raise HTTPException(status_code=500, detail="Database error while storing the document.") from exc

    logger.info("Uploaded '%s' -> %d chunks (document_id=%d)", file.filename, len(chunks), document.id)

    return UploadResponse(
        message="Document uploaded successfully",
        filename=file.filename,
        chunks=len(chunks),
        document_id=document.id,
    )
