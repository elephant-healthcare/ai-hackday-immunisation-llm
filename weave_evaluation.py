import asyncio
from textwrap import dedent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import pandas as pd
from datasets import Dataset

import weave
from weave import Scorer

import ragas
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness

import query_rag_llm

# https://medium.com/towards-data-science/productionizing-a-rag-app-04c857e0966e#fdde
judge_model = ChatOpenAI(model="gpt-4o")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
judge_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

class RagasScorer(Scorer):
    judge_model_name: str = "gemini-1.5-pro"
    embeddings_model_name: str = "models/text-embedding-004"

    # ScorerLike, with arguments matching the input dataset
    def evaluate_with_ragas(self, query, output, reference):
        def extract_llama_index_output(output):
            return {
                "question": [query],
                "contexts": [[node.node.get_content() for node in output.source_nodes]],
                "answer": [output.response],
                "reference": [reference]
        }

        judge_model = ChatGoogleGenerativeAI(model=self.judge_model_name)
        embeddings_model = GoogleGenerativeAIEmbeddings(model=self.embeddings_model_name)

        ragas_evaluation_result = ragas.evaluate(
            dataset=Dataset.from_dict(extract_llama_index_output(output)),
            metrics=[
                AnswerCorrectness(), # answer vs reference
                AnswerRelevancy(), # answer vs query
                Faithfulness(), # reponse vs retrieved context
            ], 
            llm=judge_model,
            embeddings=embeddings_model
        )
        return ragas_evaluation_result

    @weave.op
    def score(self, query, output, reference):
        # a shame to ditch the other metrics
        return self.evaluate_with_ragas(query, output, reference).scores[0]

# https://weave-docs.wandb.ai/tutorial-rag#optional-defining-a-scorer-class
class AnswerCorrectnessScorer(Scorer):
    judge_model_name: str = "gemini-1.5-pro"

    @weave.op
    def score(self, query, output, reference):
        context_precision_prompt = dedent("""
            You are evaluating the correctness of answers from a health related chatbot to patient's questions. Return a score between 1-5 measuring the correctness of the provided answer to a question, compared to the reference answer.
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
        judge_model = ChatGoogleGenerativeAI(model=self.judge_model_name)

        response = judge_model.invoke(prompt)
        return float(response.content)


if __name__ == "__main__":
    import os
    os.environ["WEAVE_PARALLELISM"] = "2"
    weave.init('ai_nurse_offline_evaluation')

    # Unfortunatelly, weave.op fails to serialise llama_index's PydanticResponse
    # https://github.com/run-llama/llama_index/blob/286ba2f60df9eac5cac3de96a31e7572d68188b0/llama-index-core/llama_index/core/base/response/schema.py#L45
    # Ignoring these events is a bit of a hack, but most of the imformation already captured in the top level predict op
    import llama_index.core
    from llama_index.core.callbacks.schema import CBEventType
    llama_index.core.global_handler.event_starts_to_ignore = [CBEventType.SYNTHESIZE, CBEventType.QUERY]

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

    mistral_model = query_rag_llm.RagModel(name="mistral", chat_llm="mistral-large-latest", embedding_model="mistral-embed")
    mistral_small_model = query_rag_llm.RagModel(name="mistral-small", chat_llm="mistral-small-latest", embedding_model="mistral-embed")
    gpt4o_mini_model = query_rag_llm.RagModel(name="gpt4o", chat_llm="gpt-4o-mini", embedding_model="text-embedding-3-small")
    gpt4o_model = query_rag_llm.RagModel(name="gpt4o", chat_llm="gpt-4o", embedding_model="text-embedding-3-small")
    gpt4o_structured_model = query_rag_llm.RagModelStructuredOutput(
        name="gpt4o-structured-output",
        chat_llm="gpt-4o", embedding_model="text-embedding-3-small",
        prompt_template= query_rag_llm.QA_STRUCTURED_PROMPT,
    )

    evaluation = weave.Evaluation(
        name=dataset_name,
        dataset=dataset,
        scorers=[RagasScorer(), AnswerCorrectnessScorer()]
    )
    asyncio.run(evaluation.evaluate(mistral_small_model))
    raise
    asyncio.run(evaluation.evaluate(mistral_model))
    asyncio.run(evaluation.evaluate(gpt4o_model))
    asyncio.run(evaluation.evaluate(gpt4o_mini_model))
    asyncio.run(evaluation.evaluate(gpt4o_structured_model))
