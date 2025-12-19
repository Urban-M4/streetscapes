from pydantic import BaseModel


class BFMSRequestSchema(BaseModel):
    image: bytes

class BFMSResponseSchema(BaseModel):
    mask: bytes
