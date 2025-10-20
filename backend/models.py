
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector aligned to training pipeline")

class PredictBatchRequest(BaseModel):
    batch: list[list[float]] = Field(..., description="List of feature vectors")
