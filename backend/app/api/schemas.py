import json
import ujson

from pickle import FALSE
from pydoc import describe
from typing import Optional, Union, List
from pydantic import BaseModel, Field

class ModelBase(BaseModel):
    name: str
    model_class: str
    classification: bool

    class Config:
        json_loads = ujson.loads


class RidgeModel(ModelBase):
    alpha: float
    tol: Optional[float] = 1e-3
    solver: Optional[str] = 'auto'
    random_state: Optional[int] = None

class LassoModel(ModelBase):
    alpha: float
    max_iter: int
    tol: Optional[float] = 1e-4
    random_state: Optional[int] = None
    selection: Optional[str] = 'cyclic'


class DecisionTreeModel(ModelBase):
    criterion: str
    splitter: Optional[str] = "best"
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = 2
    min_samples_leaf: Optional[int] = 1
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None

class Prediction(BaseModel):
    prediction: List[Union[str, float]]
