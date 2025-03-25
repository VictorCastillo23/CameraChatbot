from langchain.agents import Tool
from tools import add, subtract, multiply, divide

tools = [
    Tool(
        name = "Add",
        func = lambda x: add(*map(float, x.split())),
        description="Add two numbers. Enter the numbers separated by a space, for example: '5 7'."
    ),
    Tool(
        name = "Subtract",
        func = lambda x: subtract(*map(float, x.split())),
        description="Subtract two numbers. Enter the numbers separated by a space, for example: '10 3'."
    ),
    Tool(
        name = "Multiply",
        func = lambda x: multiply(*map(float, x.split())),
        description="Multiply two numbers. Enter the numbers separated by a space, for example: '4 6'."
    ),
    Tool(
        name = "Divide",
        func = lambda x: divide(*map(float, x.split())),
        description="Divide two numbers. Enter the numbers separated by a space, for example: '9 3'."
    )
]