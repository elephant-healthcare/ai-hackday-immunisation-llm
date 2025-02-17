import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_recall,
    context_precision,
)

from query_rag_llm import create_rag_llm


test_set_df = pd.read_json("datasets/immunisations_questions.json")

def generate_eval_df(query_engine, test_set_df=test_set_df):
    responses = [query_engine.query(question) for question in test_set_df["user_input"]]

    eval_df = test_set_df.assign(
        answer=[response.response for response in responses],
        contexts=[[node.node.get_content() for node in response.source_nodes] for response in responses],
    )
    
    evaluation_result = evaluate(
        dataset=Dataset.from_pandas(eval_df),
        metrics=[faithfulness, answer_correctness, context_recall, context_precision],
    )

    return evaluation_result.to_pandas()


if __name__ == "__main__":

    rag_llm = create_rag_llm()
    eval_df = generate_eval_df(rag_llm)

    print(eval_df)