from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> str:
    """Suma de dos números."""
    return f"The result of {a} + {b} is {a + b}."

@tool
def subtract(a: float, b: float) -> str:
    """Resta de dos números."""
    return f"The result of {a} - {b} is {a - b}."

@tool
def multiply(a: float, b: float) -> str:
    """Multiplicación de dos números."""
    return f"The result of {a} * {b} is {a * b}."

@tool
def divide(a: float, b: float) -> str:
    """División de dos números."""
    if b == 0:
        return f"Cannot be divided by 0."
    return f"The result of {a} / {b} is {a / b}."