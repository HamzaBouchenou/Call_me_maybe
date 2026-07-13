from pydantic import BaseModel
from typing import Any, Dict


class FunctionParam(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam


class Prompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
