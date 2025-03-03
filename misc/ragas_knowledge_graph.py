# https://docs.ragas.io/en/latest/howtos/customizations/testgenerator/_testgen-customisation/#setup-extractors-and-relationship-builders
# https://docs.ragas.io/en/latest/howtos/customizations/testgenerator/_testgen-customisation/#create-multi-hop-query

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def create_knowledge_graph(docs):
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    return kg

from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
apply_transforms,    default_transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
    #NERExtractor,
    #ThemesExtractor,
)

headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)

# Interesting but seems to create a lot of self relationships
relation_builder = OverlapScoreBuilder(
    property_name="keyphrases",
    new_property_name="overlap_score",
    threshold=0.01,
    distance_threshold=0.9,
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
    #relation_builder,
]


if __name__ == "__main__":
    DOCS_DIR = "./docs/curated"
    docs = DirectoryLoader(DOCS_DIR, glob="*.md").load()
    kg = create_knowledge_graph(docs)

    # Headline, Summary, Embedding, Theme, NER
    transforms = default_transforms(docs, llm, embeddings)

    apply_transforms(kg, transforms=transforms)
    kg.save("./datasets/knowledge_graph_default_transforms.json")
