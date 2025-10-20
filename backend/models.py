from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict

class PredictRequest(BaseModel):
    # EITHER provide features list (exact order), OR raw dict (named features)
    features: Optional[List[float]] = Field(None, description="Ordered feature vector")
    raw: Optional[Dict[str, float]] = Field(None, description="Named raw features dict")

    @model_validator(mode="after")
    def _either_features_or_raw(cls, v):
        if v.features is None and v.raw is None:
            raise ValueError("Provide either 'features' (list) or 'raw' (dict).")
        if v.features is not None and v.raw is not None:
            raise ValueError("Provide only one of 'features' or 'raw', not both.")
        return v

class PredictBatchRequest(BaseModel):
    batch: List[List[float]] = Field(..., description="List of ordered feature vectors")
