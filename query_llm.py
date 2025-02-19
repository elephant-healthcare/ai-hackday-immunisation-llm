from openai import OpenAI
import os
from llama_index.llms.openai import OpenAI
import weave


@weave.op()
def query_llm(query_text, priming_prompt="", model="gpt-4o", temperature=0.2):
    """Query the LLM with an optional priming prompt and return the response."""
    llm = OpenAI(model=model, temperature=temperature, api_key=os.environ["OPENAI_API_KEY"])
    full_prompt = f"{priming_prompt}\n{query_text}"
    response = llm.complete(full_prompt)
    return response

