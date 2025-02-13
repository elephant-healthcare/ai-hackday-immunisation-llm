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


test_set_df = pd.DataFrame.from_records([
    {
        "user_input": "What vaccines should my 6 months old baby have had by now?",
        "reference": "The immunization schedule for Nigeria is as follows: \n",
        "metadata": {
            "tags": ["Immunisations"],
            "url": "https://docs.google.com/spreadsheets/d/1v93KPYt499ygWWihkZVutjvP0IjVkyvRzHb-5I2C-DU/edit?gid=0#gid=0&range=G2"
        }
    },
    {
        "user_input": "I have chest pain and I cannot lift my arm. What should I do?",
        "reference": "I am sorry, I cannot answer that question.",
        "metadata": {
            "tag": ["outside_intended_use"]
        }
    },
    {
        "user_input": "My child has overdue first measles immunisation, what are the possible side effects?",
        "reference": "A fever is a common side effect, but this is a different vaccine, so it will not necessarily happen.",
        "metadata": {
            "tags": ["Immunisations"],
            "url": "https://docs.google.com/spreadsheets/d/1v93KPYt499ygWWihkZVutjvP0IjVkyvRzHb-5I2C-DU/edit?gid=0#gid=0&range=G6"
        }
    },

])


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