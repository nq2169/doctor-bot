GENERIC_PROMPT_TPL = '''
When you are asked about your identity, you must answer with 'I am a medical consultation robot built by you.'
Please answer user questions in English.
-----------
User input: {query}
'''

RETRIVAL_PROMPT_TPL = '''
Please answer the user's question based on the following search results. No supplementary or associative content is required.
If there is no relevant information in the search results, reply "Don't know".
Please answer user questions in English.
----------
Output：{query_result}
----------
User input：{query}
'''

NER_PROMPT_TPL = '''
1. Extract entity content from the following user input.
2. Note: Extract content based on the facts entered by the user. Do not make inferences or add information.

{format_instructions}
------------
User Input：{query}
------------
Output：
'''

GRAPH_PROMPT_TPL = '''
Translate the user's question from Chinese to English.
Perform a search using the translated question.
Answer the user's question by translating the relevant information from the search results back into English.
If there is no relevant information in the search results, reply with "I don't know."
----------
Output：
{query_result}
----------
User Input：{query}
'''

SEARCH_PROMPT_TPL = '''
Please answer user questions based on the following search results, and do not diverge or associate the content.
If there is no relevant information in the search results, reply "Don't know".
Please answer user questions in English.
----------
Output：{query_result}
----------
User Input：{query}
'''

SUMMARY_PROMPT_TPL = '''
Please combine the following historical conversation information and user input to summarize a concise and complete user message.
Give the summarized message directly, no other information is needed, and appropriately complete the subject and other information in the sentence.
If it is not related to the historical conversation message, the user's original message will be output directly.
Note that it only supplements the content and cannot change the semantics and sentence structure of the original message.

For Example：
-----------
Historical Conversation：
Human:What causes rhinitis?？\nAI:Rhinitis is usually caused by an infection.
User Input：What medicine should I take to get better quickly?？
-----------
Output：I got rhinitis，What medicine should I take to get better quickly?？

-----------
Historical Conversation：
{chat_history}
-----------
User Input：{query}
-----------
Output：
'''