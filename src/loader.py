from pydantic import ValidationError
from .models import FunctionDef, Prompt
import json


class InputError(Exception):
    """Raised when an input file is missing, malformed ..."""


def load_functions_definition(path: str) -> list[FunctionDef]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise InputError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise InputError(f"Invalid JSON format in: {path}")

    if not isinstance(data, list):
        raise InputError(f"'{path}' must contain a JSON array")
    if not data:
        raise InputError(f"'{path}' cannot be empty")

    parsed_functions = []
    current_names = set()
    allowed = {"name", "description", "parameters", "returns"}

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputError(f"Item {i} must be JSON object")

        item_keys = set(item.keys())

        if item_keys - allowed:
            raise InputError(
                f"Item {i} has extra unexpected keys : {item_keys - allowed}")
        if allowed - item_keys:
            raise InputError(
                f"Item {i} is missing required keys: {allowed - item_keys}")

        try:
            func = FunctionDef.model_validate(item)
        except ValidationError as e:
            raise InputError(f"Invalid function definition found : {e}") from e
        if func.name in current_names:
            raise InputError(f"Function: {func.name} stated more than once.")
        current_names.add(func.name)

        parsed_functions.append(func)

    return parsed_functions


def load_prompts(path: str) -> list[Prompt]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise InputError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise InputError(f"Invalid JSON format in: {path}")

    if not isinstance(data, list):
        raise InputError(f"'{path}' must contain a JSON array")
    if not data:
        raise InputError(f"'{path}' cannot be empty")

    parsed_prompts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputError(f"Item {i} must be JSON object")
        if set(item.keys()) != {"prompt"}:
            raise InputError(f"Item {i} must only contain the key 'prompt'")

        try:
            prompt = Prompt.model_validate(item)
        except ValidationError as e:
            raise InputError(f"Invalid prompt found at index {i}: {e}") from e

        parsed_prompts.append(prompt)
    return parsed_prompts
