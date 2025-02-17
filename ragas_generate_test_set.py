from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset.persona import Persona
from ragas.testset import TestsetGenerator

personas = [
    Persona(
        name="worried parent",
        role_description="A worried parent who is concerned about their child's health and well-being",
    ),
]
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
)

from langchain_community.document_loaders import DirectoryLoader

DOCS_DIR = "docs"
docs = DirectoryLoader(DOCS_DIR, glob="**/*.md").load()

dataset = generator.generate_with_langchain_docs(
    docs[:],
    testset_size=5,
)