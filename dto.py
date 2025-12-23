from typing import Optional

from pydantic import BaseModel


class TrainDTO(BaseModel):
    epochs: Optional[int]
    batch_size: Optional[int]

    class Config:
        extra = "allow"
