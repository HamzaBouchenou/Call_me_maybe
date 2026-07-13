import argparse
from src.parser import load_functions_definition, load_prompts
from src.prompt_builder import build_prompt
from llm_sdk.llm_sdk import Small_LLM_Model


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
        default="data/output/function_calling_tests_output.json"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
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
    for prompt in prompts:
        full_prompt = build_prompt(functions, prompt.prompt)
        print(f"Full prompt:\n{full_prompt}\n")

    llm = Small_LLM_Model()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
