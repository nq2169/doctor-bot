from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from py2neo import Graph
from config import *

import os
from dotenv import load_dotenv
load_dotenv()

def get_embeddings_model():
    model_map = {
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDINGS_MODEL')
        )
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))

def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))

def structured_output_parser(response_schemas):
    text = '''
    Please extract the entity information from the following text and output it in json format. json contains "```json" and "```" at the beginning and end.
    The following are the schema meanings and types. The output json is required to contain all the following fields:\n
    '''
    for schema in response_schemas:
        text += schema.name + ' schema, description：' + schema.description + '，type：' + schema.type + '\n'
    return text

def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string

#Connect to Neo4j
def get_neo4j_conn():
    graph = Graph(
        os.getenv('NEO4J_URI'), 
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
    return graph

if __name__ == '__main__':
    llm_model = get_llm_model()
    print(llm_model.invoke("what is disease?"))
