from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.prompts import QueryAnswerGenerationPrompt

import weave
from weave import Dataset
import json

from .ragas_knowledge_graph import create_knowledge_graph

# Initialize Weave with your project name
weave.init("ragas_generate_test_set")


llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas.testset import Testset, TestsetSample


GENERATE_QUERY_REFERENCE_PROMPT_INSTRUCTION = """
You are a copyrighter curating frequently asked questions and their answers based on available context and the persona of the audience asking questions.

1. **Generate a Query**: Based on the providedcontext, persona, term, style, and length, create a question that deeply aligns with the persona's perspective and how they would ask a question about the term.

2. **Generate an Answer**: Construct a concise and factual answer to the query, ignore the query's style use a standard english style. Using only the content from the provided context, and do not add any information not included in or not inferable from the context.
"""

# use specific entities from the KG as "Themes" for Persona-Themes matching
# Only use nodes (not relationships)
# Slightly wasteful as prompts for each nodes' themes...
generate_query_reference_prompt = QueryAnswerGenerationPrompt()
generate_query_reference_prompt.instruction = GENERATE_QUERY_REFERENCE_PROMPT_INSTRUCTION
theme_based_synthesizer = SingleHopSpecificQuerySynthesizer(
    property_name="themes", 
    generate_query_reference_prompt=generate_query_reference_prompt,
    llm=llm,
)

# Out-of-the box entities picks up too many non health related entities
entity_based_synthesizer = SingleHopSpecificQuerySynthesizer(
    property_name="entities", 
    generate_query_reference_prompt=generate_query_reference_prompt,
    llm=llm, 
)

worried_mother = Persona(
        name="worried mother",
        role_description="A worried Nigeria mother concerned about their child's health and well-being. She is not interested in politics or history. She is particularly concerned about her child's immunisation schedule and whether it is up to date, as well as possible side effects.",
    )

vaccine_hesitant_parent = Persona(
        name="vaccine hesitant parent",
        role_description="A Nigerian young parent, who is vaccine hesitant and suspicious of modern medicine in general. You are mostly ignorant of immunisation benefits and misinterpret official communications or trust conspiracy theories instead"
    )

run_config=RunConfig(
        timeout=60,
        max_retries=10,
        max_wait = 180, # default: 60
        max_workers= 2, # default: 16 <--- I think this is the setting that ensures that there are no rate limit exceptions!
    )

if __name__ == "__main__":

    import asyncio
    kg = KnowledgeGraph.load("./datasets/knowledge_graph_nutrition.json")
    #kg = create_knowledge_graph(nodes=[])
    # Doesn't return scenarios, just samples :(
    testset_generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=[worried_mother, vaccine_hesitant_parent],
    )

    testset = testset_generator.generate(
        testset_size=20,
        query_distribution=[
            (theme_based_synthesizer, 1.0),
            #(entity_based_synthesizer, 0.5),
        ],
        run_config=run_config,
    )
    testset.to_jsonl("datasets/ragas_theme_based_testset_20.jsonl")
    
    synthesizer = entity_based_synthesizer
    synthesizer = theme_based_synthesizer
    scenarios = asyncio.run(synthesizer.generate_scenarios(
        n=20,
        knowledge_graph=KnowledgeGraph(synthesizer.get_node_clusters(kg)),     # used by test generator under the hood to use chunks instead of nodes
        persona_list=[worried_mother],
        )
    )

    samples = [asyncio.run(synthesizer.generate_sample(s)) for s in scenarios]
    # No metadata to store more than the synthesizer name...
    testset = Testset(samples=[
        TestsetSample(
            eval_sample=sample,
            synthesizer_name=synthesizer.name) for sample in samples
    ])

    import json
    
    # Write enhanced JSONL with metadata - one JSON object per line
    with open("datasets/ragas_entity_based_testset_20.jsonl", "w") as f:
        f.writelines(
            sample_with_scenario_to_jsonl(sample, scenario) for sample, scenario in zip(samples, scenarios)]
        )

def sample_with_scenario_to_jsonl(sample, scenario):
    return json.dumps(
                dict(
                    **sample.to_dict(), 
                    metadata=dict(
                    scenario=dict(
                        synthesizer=synthesizer.name,
                        style=scenario.style,
                        length=scenario.length,
                        persona=scenario.persona.name,
                        term=scenario.term,
                    ),
                    synthesizer_name=synthesizer.name))
            ) + "\n"