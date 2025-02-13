from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

similarity_top_k = 2

DOCS_DIR = "./docs"

def create_rag_llm():
    docs = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        required_exts=".md"
        ).load_data()

    vector_index = VectorStoreIndex.from_documents(docs, embed_model=OpenAIEmbedding())
    rag_llm =   vector_index.as_query_engine(similarity_top_k=similarity_top_k)
    return rag_llm

if __name__ == "__main__":
    rag_llm = create_rag_llm()
    response = rag_llm.query("What is the immunization schedule for Nigeria?")
    print(response)