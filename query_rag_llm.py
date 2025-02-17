import functools
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response

import weave

import intent_classifier

QA_PROMPT = (
    "clinical guidelines:\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "You are a helpful AI nurse answering questions about immunization and maternal and child health (MCH) topics, based on the guidelines provided above."
    "MCH refers to the health of children and mothers, including pregnancy, childbirth, and postpartum care."
    "If the question is not related to immunization or MCH, answer with 'I'm sorry, I can only answer questions about immunization and MCH.'"
    "If the answer to the question cannot be clearly inferred from the guidelines, answer with 'I'm sorry, I can't answer that question based on my current clinical knowledge.'"
    "Question: {query_str}\n")

QA_SHORTER_PROMPT = (
    "<clinical_guidelines>\n"
    "{context_str}\n"
    "</clinical_guidelines>\n"
    "You are a helpful AI nurse answering patient's health questions based on the guidelines provided above."
    "Phrase your answer in a way that is easy to understand, exluding any jargon or clinical details unless specifically asked.\n"
    "Respond immediately to the question without any introduction or greetings, and conclude with a reference to the source of the answer.\n"
    "If the answer to the question cannot be clearly inferred from the guidelines, answer with 'I'm sorry, I can't answer that question based on my current clinical knowledge.'\n"
    "<question>\n{query_str}\n"
    "</question>\n"
    )

DOCS_DIR = "./docs"

class RagModel(weave.Model):
    chat_llm: str
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.1
    similarity_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
    prompt_template: str = QA_SHORTER_PROMPT

    @weave.op()
    def predict(self, query: str):
        # TODO: https://weave-docs.wandb.ai/guides/tracking/feedback/#retrieve-the-call-uuid
        intent = intent_classifier.classify_intent(query)
        if intent == intent_classifier.MALICIOUS_LABEL:
            return Response("I'm sorry, I can't answer that question.")
 
        # Annoyingly, not finding yet the right pattern to initialize the query engine
        # once in the ctor, so relying on the memoised call here.
        query_engine = get_query_engine(
            data_dir=DOCS_DIR,
            chat_llm=self.chat_llm,
            embedding_model=self.embedding_model,
            temperature=self.temperature,
            similarity_top_k=self.similarity_top_k,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            prompt_template=self.prompt_template,
        )
        response = query_engine.query(query)
        return response
    

@functools.lru_cache(maxsize=1)
def get_query_engine(
    data_dir: str,
    chat_llm: str,
    embedding_model: str,
    temperature: float,
    similarity_top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    prompt_template: str,
):
    documents = SimpleDirectoryReader(data_dir, required_exts=[".md"]).load_data()
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=OpenAIEmbedding(model=embedding_model))
    llm = MistralAI(model=chat_llm, temperature=temperature) if "mistral" in chat_llm else OpenAI(temperature=temperature, model=chat_llm)
    qa_template = PromptTemplate(prompt_template)

    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        llm=llm,
        text_qa_template=qa_template,
    )

def create_query_rag_llm_v2(chat_llm = "mistral-large-latest", embedding_model="text-embedding-3-small"):
    model = RagModel(name=chat_llm, chat_llm=chat_llm, embedding_model=embedding_model)
    return model.predict


SIMILARITY_TOP_K = 3
RESPONSE_MODE = "compact"

def create_rag_llm():
    docs = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        required_exts=".md",
        ).load_data()

    vector_index = VectorStoreIndex.from_documents(docs, embed_model=OpenAIEmbedding())
    rag_llm = vector_index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode=RESPONSE_MODE,
        text_qa_template=PromptTemplate(QA_PROMPT),
    )
    return rag_llm

@weave.op()
def query_rag_llm(query_str):
    rag_llm = create_rag_llm()
    response = rag_llm.query(query_str)
    return response

if __name__ == "__main__":
    query_rag_llm = create_query_rag_llm_v2()
    response = query_rag_llm("My baby is 3 months old. Which immunisations should they have had by now?")
    print(response)