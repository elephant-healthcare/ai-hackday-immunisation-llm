from openai import OpenAI
import os
from llama_index.llms.openai import OpenAI
import weave

llm_name = "gpt-4o-mini"
model_temperature = 0.8

@weave.op()
def query_llm(query_text, priming_prompt=""):
    """Query the LLM with an optional priming prompt and return the response."""
    llm = OpenAI(model=llm_name, temperature=model_temperature, api_key=os.environ["OPENAI_API_KEY"])
    full_prompt = f"{priming_prompt}\n{query_text}"
    response = llm.complete(full_prompt)
    return response

