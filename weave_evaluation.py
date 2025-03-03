import asyncio
from textwrap import dedent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from datasets import Dataset

import weave

import ragas
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness

import query_rag_llm

# https://medium.com/towards-data-science/productionizing-a-rag-app-04c857e0966e#fdde

JUDGE_MODEL_NAME = "gpt-4o"
JUDGE_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# ScorerLike, with arguments matching the input dataset
def evaluate_with_ragas(query, output, reference):

    def extract_llama_index_output(output):
        return {
            "question": [query],
            "contexts": [[node.node.get_content() for node in output.source_nodes]],
            "answer": [output.response],
            "reference": [reference]
    }

    judge_model = ChatOpenAI(model=JUDGE_MODEL_NAME)
    embeddings_model = OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL_NAME)

    return ragas.evaluate(
        dataset=Dataset.from_dict(extract_llama_index_output(output)),
        metrics=[
            AnswerCorrectness(), # answer vs reference
            AnswerRelevancy(), # answer vs query
            Faithfulness(), # reponse vs retrieved context
        ], 
        llm=judge_model,
        embeddings=embeddings_model
    )

@weave.op
def ragas_scorer(query, output, reference):
    ragas_evaluation_result = evaluate_with_ragas(query, output, reference)
    return ragas_evaluation_result.scores[0]

# https://weave-docs.wandb.ai/tutorial-rag#optional-defining-a-scorer-class
@weave.op
def answer_correctness_scorer(query, output, reference):
    context_precision_prompt = dedent("""You are evaluating the correctness of answers from a health related chatbot to patient's questions. Return a score between 1-5 measuring the correctness of the provided answer to a question, compared to the reference answer.
        <question>
        {question}
        </question>
        <reference>
        {reference}
        </reference>
        <answer>
        {answer}
        </answer>
        Return only the integer correctness score without any other formatting""")
    prompt = context_precision_prompt.format(
        question=query,
        answer=output.response,
        reference=reference
    )
    # pip install -U langchain-google-vertexai
    client = ChatOpenAI(model=JUDGE_MODEL_NAME)
    response = client.invoke(prompt)
    return float(response.content)

import llama_index.core
from llama_index.core.callbacks.schema import CBEventType

# Unfortunatelly, weave.op fails to serialise llama_index's PydanticResponse
# https://github.com/run-llama/llama_index/blob/286ba2f60df9eac5cac3de96a31e7572d68188b0/llama-index-core/llama_index/core/base/response/schema.py#L45
# Ignoring these events is a bit of a hack, but most of the imformation already captured in the top level predict op
llama_index.core.global_handler.event_starts_to_ignore = [CBEventType.SYNTHESIZE, CBEventType.QUERY]


if __name__ == "__main__":
    weave.init('immunisation_questions_offline')

    dataset_name = "immunisation_plus_outside_intended_use_questions"
    try:
        dataset = weave.ref(dataset_name).get()
        test_set_questions = dataset.rows
    except:
        immunisation_test_set_df = pd.read_json("datasets/immunisations_questions.json")
        outside_intented_use_test_set_df = pd.read_json("datasets/outside_intended_use_questions.json")
        test_set_questions = immunisation_test_set_df.to_dict(orient="records") + outside_intented_use_test_set_df.to_dict(orient="records")

        dataset = weave.Dataset(
            name=dataset_name,
            description="immunisation questions related to schedules and side effects, plus outside intended use questions",
            rows=test_set_questions
        )
        weave.publish(dataset)

    gpt4o_model = query_rag_llm.RagModel(name="gpt4o", chat_llm="gpt-4o", embedding_model="text-embedding-3-small")
    mistral_model = query_rag_llm.RagModel(name="mistral-large", chat_llm="mistral-large-latest", embedding_model="text-embedding-3-small")

    gpt4o_structured_model = query_rag_llm.RagModelStructuredOutput(
        name="gpt4o-structured-output",
        chat_llm="gpt-4o", embedding_model="text-embedding-3-small",
        prompt_template= query_rag_llm.QA_STRUCTURED_PROMPT,
    )

    # query = test_set_questions[0]["query"]
    # output = gpt4o_structured_model.predict(query)
    # evaluate_with_ragas(
    #     query=query,
    #     output=output,
    #     reference=test_set_questions[0]["reference"]
    # )

    evaluation = weave.Evaluation(
        name=dataset_name,
        dataset=dataset,
        scorers=[ragas_scorer, answer_correctness_scorer]
    )
    asyncio.run(evaluation.evaluate(gpt4o_model))
    asyncio.run(evaluation.evaluate(mistral_model))
    asyncio.run(evaluation.evaluate(gpt4o_structured_model))