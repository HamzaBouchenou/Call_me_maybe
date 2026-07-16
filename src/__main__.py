import argparse
from src.loader import load_functions_definition, load_prompts
from src.prompt_builder import build_prompt
from llm_sdk import Small_LLM_Model
from src.decoder import get_valid_token_ids, load_vocab, get_next_valid_token, extract_valid_json
import json
import time


def build_parser():
    parser = argparse.ArgumentParser(description="Introduction to function "
                                     "calling in LLms.")

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    return parser


def run_pipeline(args: argparse.Namespace):
    functions = load_functions_definition(args.functions_definition)
    if not functions:
        raise RuntimeError(
            "No functions loaded. Please check the functions definition file."
        )
    prompts = load_prompts(args.input)
    if not prompts:
        raise RuntimeError(
            "No prompts loaded. Please check the prompts file."
        )

    instructions = build_prompt(functions)
    print("Loading model...")
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    valid_ids = get_valid_token_ids(vocab)
    results = []

    print("processing prompts...\n")
    for pt in prompts:
        prompt = pt.prompt
        full_prompt = f"{instructions}\nUser prompt: {prompt}\nAssistant: "
        input_ids = model.encode(full_prompt)
        prompt_generated_ids = input_ids[0].tolist()
        all_generated = []

        valid_json = False
        all_generated.extend(model.encode('{"name": "')[0].tolist())
        for _ in range(50):
            logits = model.get_logits_from_input_ids(prompt_generated_ids +
                                                     all_generated)
            next_id = get_next_valid_token(logits, valid_ids)
            all_generated.append(next_id)
            text = model.decode(all_generated)
            valid_json = extract_valid_json(text)
            if valid_json:
                try:
                    loaded = json.loads(valid_json)
                    break
                except Exception:
                    pass

        print(valid_json)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
