from typing import List, Dict, Any
import json
from llm_sdk import Small_LLM_Model
from .models import FunctionCall, FunctionDef


def load_vocab(model: Small_LLM_Model) -> Dict[str, Any]:
    """Load the vocabulary of the model."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return raw


def get_valid_token_ids(vocab: Dict[str, Any]) -> set[int]:
    """Get the list of valid token IDs from the vocabulary."""
    accepted_tks = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_-+.,:/\'?!()[]{}"ĠĊ'
    )
    valid = set()
    for token, token_id in vocab.items():
        if token and all(char in accepted_tks for char in token):
            valid.add(token_id)
    return valid


def get_next_valid_token(logits: List[float], valid_ids: set[int]) -> int:
    """Get the next valid token ID based on the logits and valid IDs."""
    return max(valid_ids, key=lambda tid: logits[tid])


def extract_valid_json(text: str) -> str | None:
    """Extract valid JSON from the given text."""
    start = text.find('{')
    if start == -1:
        return None

    count_braces = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count_braces += 1
        elif text[i] == '}':
            count_braces -= 1

        if count_braces == 0:
            return text[start:i + 1]
    return None
