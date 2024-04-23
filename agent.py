from utils import *
from config import *
from prompt import *

import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from py2neo import *
from py2neo import Graph
from neo4j import GraphDatabase
from translate import Translator


class Agent():
    def __init__(self):
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'), 
            embedding_function = get_embeddings_model()
        )


    def generic_func(self, query):
        # Using a template from an external prompt definition
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        # Create LLMChain instance
        llm_chain = LLMChain(
            llm = get_llm_model(), 
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        # Execute the chain with the provided query
        return llm_chain.invoke(query)['text']
    
    def retrival_func(self, x, query):
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=3)
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]

        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else 'No records'
        }
        return retrival_chain.invoke(inputs)['text']
     
    #The translator is set to translate the user input into Chinese (Since the database is stored in Chinese)
    def translate_to_chinese(self, text):
        translator = Translator(to_lang="zh")
        translation = translator.translate(text)
        return translation

    def graph_func(self, x, query):
        # Naming entity recognition
        query = self.translate_to_chinese(query)
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='disease name entity'),
            ResponseSchema(type='list', name='symptom', description='symptom entity'),
            ResponseSchema(type='list', name='drug', description='drug entity'),
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)

        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables = {'format_instructions': format_instructions},
            input_variables = ['query']
        )

        ner_chain = LLMChain(
            llm = get_llm_model(),
            prompt = ner_prompt,
            verbose = os.getenv('VERBOSE')
        )

        result = ner_chain.invoke({
            'query': query
        })['text']
        
        ner_result = output_parser.parse(result)

        #print(ner_result)

        # Fill the template with results from Named Entity Recognition
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
            #print(graph_templates)
            #exit()
        if not graph_templates:
            return 
    
        # Calculate question similarity, filter for the most relevant questions
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        #print(graph_documents)
        #exit()
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)


        #Execute CQL and get result
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'Question：{question}\nAnswer：{answer_str}')
            except:
                pass
        #print(query_result)
        #exit()

        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else 'No records'
        }
        return graph_chain.invoke(inputs)
    
    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        llm_request_chain = LLMRequestsChain(
            llm_chain = llm_chain,
            requests_key = 'query_result'
        )
        inputs = {
            'query': query,
            'url': 'https://www.google.com/search?q='+query.replace(' ', '+')
        }
        return llm_request_chain.invoke(inputs)


    def query(self, query):
        tools = [
            Tool.from_function(
                name = 'generic_func',
                func = self.generic_func,
                description = 'Used to answer general questions, such as saying hello, asking who you are, etc.',
            ),
            Tool.from_function(
                name = 'retrival_func',
                func = lambda x: self.retrival_func(x, query),
                description = 'Used to anwer domain knowlege, particularly about medicine',            
            ),

            Tool(
                name = 'graph_func',
                func = lambda x: self.graph_func(x, query),
                description = 'Used to answer medical-related questions about diseases, symptoms, medications, etc.',
            ),
            Tool(
                name = 'search_func',
                func = self.search_func,
                description = 'Use search engines to answer general questions when other tools do not have the correct answer.',
            ),
        ]

        prefix = """Please answer the following questions in English to the best of your ability. You can use the following tools："""
        suffix = """Begin!

        History: {chat_history}
        Question: {input}
        Thought:{agent_scratchpad}"""

        agent_prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=['input', 'agent_scratchpad', 'chat_history']
        )

        llm_chain = LLMChain(llm=get_llm_model(), prompt=agent_prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain)

        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent = agent, 
            tools = tools, 
            memory = memory, 
            verbose = os.getenv('VERBOSE')
        )
        return agent_chain.invoke({'input': query})


#Test
if __name__ == '__main__':
    agent = Agent()
    #print(agent.generic_func('What is your name?'))
    #print(agent.graph_func('What are common symptoms of colds?'))
    #print(agent.graph_func('鼻炎和感冒是并发症吗？'))
    #print(agent.graph_func('What medicine can cure a cold quickly? Can I take amoxicillin?'))
    #print(agent.graph_func('What causes a cold?'))
    #print(agent.graph_func('What medicine should be taken for a quick relief from cold? Can Amoxicillin be taken?'))
    #print(agent.graph_func('Are colds and rhinitis complications?'))
    #print(agent.graph_func('How to treat rhinitis?'))
    #print(agent.search_func('Can orange cure a cold?'))
    print(agent.query('鼻炎是什么引起的?'))
    #print(agent.query('What is your name?'))
    #print(agent.query('Are rhinitis and colds complications?'))
    #print(agent.query('How to treat rhinitis?'))
    #print(agent.query('Can orange cure a cold?'))


    


