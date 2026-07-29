import argparse
from src.loader import load_functions_definition, load_prompts, InputError
from src.prompt_builder import build_prompt
from llm_sdk import Small_LLM_Model
from src.decoder import (build_id_to_token, load_vocab, encode_text,
                         generate_function_call)
import json
import time
import os


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
    try:
        functions = load_functions_definition(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as e:
        print(f"Caught a parsing error: {e}")
        return

    instructions = build_prompt(functions)
    print("Loading model...")
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    id_to_token = build_id_to_token(vocab)

    results = []
    print("processing prompts...\n")
    start_time = time.time()

    for pt in prompts:
        prompt = pt.prompt
        full_prompt = f"{instructions}\nUser prompt: {prompt}\nAssistant: "
        prompt_ids = encode_text(model, full_prompt)
        try:
            call = generate_function_call(
                model, id_to_token, functions, prompt_ids)
            result = {"prompt": pt.prompt, **call}
            results.append(result)
            print(f"{result}")
        except Exception as e:
            print(f"Caught error while processing : {pt.prompt}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nProcessed {len(prompts)} prompts in {elapsed:.2f}s ")
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    except PermissionError:
        print("File does not have permission")
        return
    except TypeError:
        print("Could not serialize results to JSON")
        return
   

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
