"""Constrained decoding: force known JSON structure, let the model choose
only between reachable options at each step."""

from typing import List, Dict, Any, Tuple, cast, Callable
from llm_sdk import Small_LLM_Model
from .models import FunctionDef, FunctionParam


def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
    """Load token string -> token id mapping from the model's vocab file."""
    import json
    with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
        return cast(Dict[str, int], json.load(f))


def build_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    """Reverse vocab: token id -> normalized token string."""
    return {tid: tok.replace("Ġ", " ").replace("Ċ", "\n")
            for tok, tid in vocab.items()}


def encode_text(model: Small_LLM_Model, text: str) -> List[int]:
    """Encode literal text to a flat list of token ids."""
    ids = model.encode(text)
    return cast(List[int], ids[0].tolist())


def _pick(model: Small_LLM_Model, ids: List[int], valid: set) -> int:
    """Return the highest-logit token id among `valid`."""
    if not valid:
        raise RuntimeError("No valid token ids at this generation step.")
    logits = model.get_logits_from_input_ids(ids)
    return cast(int, max(valid, key=lambda tid: logits[tid]))


def generate_constrained_choice(
    model: Small_LLM_Model,
    id_to_token: Dict[int, str],
    ids: List[int],
    options: List[str],
    max_steps: int = 30,
) -> Tuple[str, List[int]]:
    """Force generation to exactly match one of `options`, token by token."""
    matched = ""
    for _ in range(max_steps):
        valid = {
            tid for tid, tok in id_to_token.items()
            if tok and any(o.startswith(matched + tok) for o in options)
        }
        tid = _pick(model, ids, valid)
        ids = ids + [tid]
        matched += id_to_token[tid]
        if matched in options:
            return matched, ids
    raise RuntimeError(f"Could not resolve choice among {options!r}")


def generate_constrained_text(
    model: Small_LLM_Model,
    id_to_token: Dict[int, str],
    ids: List[int],
    char_ok: Callable[[str], bool],
    stop_chars: set,
    quoted: bool,
    max_len: int = 100,
) -> Tuple[str, List[int]]:
    """Generate free-form content: model picks tokens matching `char_ok`
    until a token from `stop_chars` becomes the best choice, or a token
    containing an embedded quote/stop marker signals the boundary."""
    if quoted:
        ids = ids + encode_text(model, '"')
    content: List[int] = []
    for _ in range(max_len):
        valid = {
            tid for tid, tok in id_to_token.items()
            if (tok and char_ok(tok)) or tok in stop_chars
            or (quoted and '"' in tok)
        }
        tid = _pick(model, ids + content, valid)
        tok = id_to_token[tid]

        if quoted and '"' in tok:
            before = tok.split('"', 1)[0]
            if before:
                content.append(tid)
            break
        if tok in stop_chars:
            break

        content.append(tid)

    text = model.decode(content) if content else ""
    text = text.split('"', 1)[0] if quoted else text
    if quoted:
        ids = ids + content + encode_text(model, '"')
    else:
        ids = ids + content
    return text, ids


def generate_value(
    model: Small_LLM_Model,
    id_to_token: Dict[int, str],
    ids: List[int],
    param: FunctionParam,
) -> Tuple[Any, List[int]]:
    """Generate a single typed parameter value."""
    if param.type == "boolean":
        text, ids = generate_constrained_choice(
            model, id_to_token, ids, ["true", "false"]
        )
        return text == "true", ids

    if param.type == "number":
        text, ids = generate_constrained_text(
            model, id_to_token, ids,
            char_ok=lambda t: all(c in set("0123456789.-") for c in t),
            stop_chars={",", "}", " ", "\n"}, quoted=False,
        )
        try:
            return float(text), ids
        except ValueError:
            return 0.0, ids

    if param.type == "integer":
        text, ids = generate_constrained_text(
            model, id_to_token, ids,
            char_ok=lambda t: all(c in set("0123456789-") for c in t),
            stop_chars={",", "}", " ", "\n"}, quoted=False,
        )
        try:
            return int(text), ids
        except ValueError:
            return 0, ids

    if param.type == "string":
        text, ids = generate_constrained_text(
            model, id_to_token, ids,
            char_ok=lambda t: '"' not in t,
            stop_chars=set(), quoted=True,
        )
        return text, ids

    raise ValueError(f"Unsupported parameter type: {param.type}")


def generate_function_call(
    model: Small_LLM_Model,
    id_to_token: Dict[int, str],
    functions: List[FunctionDef],
    prompt_ids: List[int],
) -> Dict[str, Any]:
    """Run phase-aware constrained generation: force all known structure,
    constrain the model only on the function name and each param value."""
    ids = prompt_ids + encode_text(model, '{"name": "')
    name, ids = generate_constrained_choice(
        model, id_to_token, ids, [f'"{f.name}"' for f in functions]
    )
    name = name.strip('"')
    chosen = next((f for f in functions if f.name == name), None)
    if chosen is None:
        raise RuntimeError(
            f"Generated name {name!r} matches no known function."
        )

    ids = ids + encode_text(model, '", "parameters": {')
    parameters: Dict[str, Any] = {}
    items = list(chosen.parameters.items())
    for i, (pname, pdef) in enumerate(items):
        ids = ids + encode_text(model, f'"{pname}": ')
        value, ids = generate_value(model, id_to_token, ids, pdef)
        parameters[pname] = value
        if i < len(items) - 1:
            ids = ids + encode_text(model, ", ")

    return {"name": name, "parameters": parameters}
