from .models import FunctionDef
from typing import List


def build_prompt(functions: List[FunctionDef]) -> str:

    instructions = [
        "You are a function-calling assistant.",
        "Choose the correct function from the list below.",
        "Return ONLY a JSON object with name and parameters",
        "Do not include any extra text or explanations.",
        "Available functions:"
    ]
    for func in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in func.parameters.items()
        )
        instructions.append(
            f"- {func.name}({params}): {func.description}"
        )
    return "\n".join(instructions)