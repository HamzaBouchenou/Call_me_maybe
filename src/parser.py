from pydantic import ValidationError
from .models import FunctionDef, Prompt
import json


def load_functions_definition(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [FunctionDef(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in: {path}")
    except ValidationError:
        raise RuntimeError(f"Schema mismatch in: {path}")


def load_prompts(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in: {path}")
    except ValidationError:
        raise RuntimeError(f"Schema mismatch in: {path}")
