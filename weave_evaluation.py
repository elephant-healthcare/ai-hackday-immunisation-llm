import asyncio
from textwrap import dedent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from datasets import Dataset

import weave

import ragas
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness

from query_rag_llm import RagModel

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


if __name__ == "__main__":
    weave.init('immunisation_questions_offline')

    dataset_name = "immunisation_questions"
    try:
        dataset = weave.ref(dataset_name).get()
    except:
        test_set_df = pd.read_json("datasets/immunisations_questions.json")

        dataset = weave.Dataset(
            name="immunisation_questions",
            description="immunisation questions related to schedules and side effects",
            rows=test_set_df.to_dict(orient="records")
        )
        weave.publish(dataset)

    gpt4o_model = RagModel(name="gpt4o", chat_llm="gpt-4o", embedding_model="text-embedding-3-small")
    mistral_model = RagModel(name="mistral-large", chat_llm="mistral-large-latest", embedding_model="text-embedding-3-small")

    # output = mistral_model.predict(test_set_df.loc[0, "query"])
    # evaluate_with_ragas(
    #    test_set_df.loc[0, "query"],
    #    mistral_model.predict(test_set_df.loc[0, "query"]),
    #    test_set_df.loc[0, "reference"]
    #    )
    # answer_correctness(
    #    test_set_df.loc[0, "query"],
    #    output,
    #    test_set_df.loc[0, "reference"]
    #    )

    evaluation = weave.Evaluation(
        name="immunisation_questions",
        dataset=dataset,
        scorers=[ragas_scorer, answer_correctness_scorer]
    )
    asyncio.run(evaluation.evaluate(gpt4o_model))
    asyncio.run(evaluation.evaluate(mistral_model))