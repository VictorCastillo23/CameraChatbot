from langchain.agents import create_openai_tools_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tools_list import tools
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

with open("prompt.md", "r") as file:
    system_prompt = file.read()

prompt_template = PromptTemplate(
    template=system_prompt,
    input_variables=["input", "agent_scratchpad"]
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key
)

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

response_add = agent.invoke({
    "input": "Add 5 and 7.",
    "intermediate_steps": []
})
if isinstance(response_add, dict) and "output" in response_add:
    print("Resultado:", response_add["output"])
elif isinstance(response_add, list):
    print("Resultados intermedios:", response_add)
else:
    print("Respuesta completa:", response_add)

"""
response_mul = agent.invoke({"input": "Multiplica 30 y 9."})
print(response_mul)

response_div = agent.invoke({"input": "Divide 10 y 5."})
print(response_div)

response_sub = agent.invoke({"input": "Resta 5.5 y 1.5."})
print(response_sub)

response_invalid = agent.invoke({"input": "Resta 4 a nada"})
print(response_invalid)

response_div_zero = agent.invoke({"input": "Divide 10 y 0."})
print(response_div_zero)
"""