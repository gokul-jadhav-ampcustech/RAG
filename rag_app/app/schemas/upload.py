from pydantic import BaseModel


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    document_id: int
