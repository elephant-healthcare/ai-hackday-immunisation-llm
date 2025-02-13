# https://docs.ragas.io/en/latest/howtos/integrations/_llamaindex/#building-the-testset
# https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/ragas_retrieval_evals_tutorial.ipynb#scrollTo=1c74e381
# https://docs.ragas.io/en/latest/howtos/integrations/_arize/

# pip install llama-index llama-index-llms-mistralai ragas
from ragas.testset import TestsetGenerator



from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


docs = SimpleDirectoryReader(
    input_dir="docs/nyc",
    ).load_data()

# Dataset generation
generator = TestsetGenerator.from_llama_index(
    llm=OpenAI(model="gpt-4o-mini"),
    embedding_model=OpenAIEmbedding(model="text-embedding-3-small"),
)

dataset = generator.generate_with_llamaindex_docs(docs, testset_size=10)
test_df = dataset.to_pandas()
test_df.head()

phoenix_client = px.Client()
dataset = phoenix_client.upload_dataset(
    dataframe=test_df,
    dataset_name="nyc-ragas-dataset",
    input_keys=["user_input"],
    output_keys=["reference"],
)


# https://docs.arize.com/phoenix/quickstart
# pip install openinference-instrumentation-openai openai
# pip install 'arize-phoenix==7.10.3'

import os
from phoenix.otel import register
PHOENIX_API_KEY = os.environ['PHOENIX_API_KEY']
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# configure the Phoenix tracer
tracer_provider = register() 


#https://docs.ragas.io/en/latest/howtos/integrations/_arize/#4-generate-your-synthetic-test-dataset
# https://docs.llamaindex.ai/en/stable/module_guides/observability/

import phoenix as px
from phoenix.trace import using_project
from llama_index.core import set_global_handler

# Launch local app, not needed as using the cloud API
# session = px.launch_app()
set_global_handler("arize_phoenix")

# From collab, duplcated with the above?
# Attempting to instrument while already instrumented
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

#
# https://docs.ragas.io/en/latest/howtos/integrations/_arize/#5-build-your-rag-application-with-llamaindex
#
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

vector_index = VectorStoreIndex.from_documents(docs, embed_model=OpenAIEmbedding())
query_engine = vector_index.as_query_engine(similarity_top_k=2)

from tqdm.auto import tqdm

def generate_response(question):
    response = query_engine.query(question)
    return {
        "answer": response.response,
        "contexts": [c.node.get_content() for c in response.source_nodes],
    }

responses = [generate_response(q) for q in tqdm(test_df["user_input"].values)]
ragas_evals_df = test_df.assign(
    answer=[response["answer"] for response in responses],
    contexts=[response["contexts"] for response in responses],
)

ragas_evals_df.head(2)

# 
from phoenix.session.evaluation import get_qa_with_reference

# pip install openinference-instrumentation-langchain
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()
# dataset containing span data for evaluation with Ragas
spans_dataframe = get_qa_with_reference(client)
spans_dataframe.head()


from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()
#Evaluate your LLM traces and view the evaluation scores in dataframe format.


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

ragas_eval_dataset = Dataset.from_pandas(ragas_evals_df.rename(columns={'user_input': 'question', 'reference': 'ground_truth'}))

evaluation_result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=[faithfulness, answer_correctness, context_recall, context_precision],
)
eval_scores_df = pd.DataFrame(evaluation_result.scores)

# TBC!
# https://app.phoenix.arize.com/projects/UHJvamVjdDox
# Submit your evaluations to Phoenix so they are visible as annotations on your spans.
# https://docs.ragas.io/en/latest/howtos/integrations/_arize/#6-evaluate-your-llm-application
