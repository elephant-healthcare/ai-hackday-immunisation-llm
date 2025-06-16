import functools
from typing import Optional
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
# pip install llama-index-embeddings-mistralai
from llama_index.embeddings.mistralai import MistralAIEmbedding
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

from pydantic import BaseModel

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

# https://chatgpt.com/share/e/67c475a8-fd08-8006-a36e-f1deea5f31ff
QA_STRUCTURED_PROMPT = (
    "<clinical_guidelines>\n"
    "{context_str}\n"
    "</clinical_guidelines>\n"
    "You are a helpful AI nurse answering patient's health questions based on the guidelines provided above."
    "Phrase your answer in a way that is easy to understand, exluding any jargon or clinical details unless specifically asked.\n"
    "Respond immediately to the question without any introduction or greetings, and conclude with a reference to the source of the answer.\n"
"""Format the output with the following fields:
 - answer: a concise answer exclusively based on the provided context. The lower the context sufficiency, the shorter should be the answer and the quicker you should just suggest to refer to a healthcare professional.
 - context_sufficiency: a score ranging from 0 to 5 grading how complete is the provided context to answer the question, where 0 would be a question completly off topic from the context, 2 a question partially covered but missing crucial information to adress key parts of the question, 4 a question almost entirely covered except for secondary details and 5 a question fully answered without any ambiguity using the context.
 - missing_information_rationale: the rationale for low context sufficiency scores and what piece of information is missing information.
 - missing_information_keywords: the specific pieces of information in a key word format, such as "immunisation schedule".
"""
    "<question>\n{query_str}\n"
    "</question>\n"
)

# TODO: Migrate fields specification from based prompt into Pydantic model fields
class QueryEngineOutput(BaseModel):
    answer: str
    context_sufficiency: int
    missing_information_rationale: str
    missing_information_keywords: list[str]

DOCS_DIR = "./docs/curated"

class RagModel(weave.Model):
    chat_llm: str
    embedding_model: str = "mistral-embed"
    temperature: float = 0.1
    similarity_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
    prompt_template: str = QA_SHORTER_PROMPT

    @weave.op()
    def predict(self, query: str):
        # TODO: https://weave-docs.wandb.ai/guides/tracking/feedback/#retrieve-the-call-uuid
        intent = intent_classifier.classify_intent(query, model=self.chat_llm)
        if intent == intent_classifier.MALICIOUS_LABEL:
            return Response("I'm sorry, I can't answer that question. This doesn't replace a human nurse.")
 
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


class RagModelStructuredOutput(weave.Model):
    chat_llm: str
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.1
    similarity_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
    prompt_template: str = QA_STRUCTURED_PROMPT
    # Messes up with Weave serialisation I think
    #query_engine_output_cls: type[Optional[BaseModel]] = QueryEngineOutput

    @weave.op()
    def predict(self, query: str):
        # TODO: https://weave-docs.wandb.ai/guides/tracking/feedback/#retrieve-the-call-uuid
        intent = intent_classifier.classify_intent(query, model=self.chat_llm)
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
            query_engine_output_cls=QueryEngineOutput,
        )
        structured_response = query_engine.query(query)
        final_answer = structured_response.answer if structured_response.context_sufficiency > 2 else "I cannot answer that question based on my current clinical knowledge."

        # Especially needed as weave.op seems to choke on PydanticResponse
        response = Response(
            response= final_answer,
            source_nodes=structured_response.source_nodes,
            metadata=dict(structured_response.metadata,
                relevant_contexts=structured_response.relevant_contexts,
                context_sufficiency=structured_response.context_sufficiency,
                missing_information_rationale=structured_response.missing_information_rationale,
                missing_information_keywords=structured_response.missing_information_keywords,
            )
        )
        return response


@functools.lru_cache(maxsize=10)
def get_query_engine(
    data_dir: str,
    chat_llm: str,
    embedding_model: str,
    temperature: float,
    similarity_top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    prompt_template: str,
    query_engine_output_cls: Optional[BaseModel] = None
):
    documents = SimpleDirectoryReader(data_dir, required_exts=[".md"]).load_data()
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    embed_model = MistralAIEmbedding(model_name=embedding_model) if "mistral" in embedding_model else OpenAIEmbedding(model=embedding_model)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    llm = MistralAI(model=chat_llm, temperature=temperature) if "mistral" in chat_llm else OpenAI(temperature=temperature, model=chat_llm)
    qa_template = PromptTemplate(prompt_template)

    return index.as_query_engine(
        output_cls=query_engine_output_cls,
        similarity_top_k=similarity_top_k,
        llm=llm,
        text_qa_template=qa_template,
        response_mode="compact",
    )

def create_query_rag_llm_v2(chat_llm = "mistral-large-latest", embedding_model="mistral-embed"):
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
    model = RagModel(
        chat_llm="mistral-large-latest", 
        embedding_model="mistral-embed"
    )

    # TODO:  ValueError: Expected at least one tool call, but got 0 tool calls. when using mistral-large-latest
    #model = RagModelStructuredOutput(
    #    chat_llm="gpt-4o", 
    #    embedding_model="text-embedding-3-small"
    #)

    query = """"I born my twins at 33wks as preterm they're 4month now and their weight is 3.5 and 3.3. I am worried that they are not gorwing enough, please what can I give them to gain weight?"""
    query = """"What vaccines should my 6 monts old have received by now?"""
    response = model.predict(query)
    print(response)