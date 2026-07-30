from pydantic import BaseModel, field_validator, model_validator
from typing import Any, Dict, Literal


ParamType = Literal["number", "integer", "string", "boolean"]


class FunctionParam(BaseModel):
    type: ParamType


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParam]
    returns: FunctionParam

    @field_validator("name")
    @classmethod
    def check_name(cls, val: str) -> str:
        """Reject empty or whitespace-only function names."""
        if not val or not val.strip():
            raise ValueError("name must not be empty")
        return val.strip()

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, val: str) -> str:
        """Reject empty or whitespace-only descriptions."""
        if not val or not val.strip():
            raise ValueError("description must not be empty")
        return val.strip()

    @model_validator(mode="after")
    def parameters_not_empty(self) -> "FunctionDef":
        """Reject functions declared with zero parameters."""
        if not self.parameters:
            raise ValueError(
                f"function '{self.name}' has no parameters declared")
        return self


class Prompt(BaseModel):
    """A single natural-language test prompt."""
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, val: str) -> str:
        """Reject empty or whitespace-only prompts."""
        if not val or not val.strip():
            raise ValueError("prompt must not be empty")
        return val.strip()


class FunctionCall(BaseModel):
    """A generated function call result, ready for output."""
    prompt: str
    name: str
    parameters: Dict[str, Any]
