from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.prompts import PromptTemplate
import weave

QA_PROMPT_TMPL = PromptTemplate(
    "clinical guidelines:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "You are a helpful AI nurse answering questions about immunization and maternal and child health (MCH) topics, based on the guidelines provided above."
    "MCH refers to the health of children and mothers, including pregnancy, childbirth, and postpartum care."
    "If the question is not related to immunization or MCH, answer with 'I'm sorry, I can only answer questions about immunization and MCH.'"
    "If the answer to the question cannot be clearly inferred from the guidelines, answer with 'I'm sorry, I can't answer that question based on my current clinical knowledge.'"
    "Question: {query_str}\n")

SIMILARITY_TOP_K = 3
RESPONSE_MODE = "compact"
DOCS_DIR = "./docs"


def create_rag_llm():
    docs = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        required_exts=".md",
        ).load_data()

    vector_index = VectorStoreIndex.from_documents(docs, embed_model=OpenAIEmbedding())
    rag_llm = vector_index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode=RESPONSE_MODE,
        text_qa_template=QA_PROMPT_TMPL,
    )
    return rag_llm

@weave.op()
def query_rag_llm(query_str):
    rag_llm = create_rag_llm()
    response = rag_llm.query(query_str)
    return response

if __name__ == "__main__":
    rag_llm = create_rag_llm()
    response = rag_llm.query("What is the immunization schedule for Nigeria?")
    print(response)