import json
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key= os.getenv('OPENAI_API_KEY')

from pypdf import PdfReader

#COMMENTS
#structured output parser for dictionary of strings
#enum output parser does not force usage of enums- only throws an error
#possible concern of commas in topics
#simply categorizing- only belongs to 1 topic
#not influenced by headers and footers but ideal for question retrieval\

topicList = []
filename = "topics.txt"
with open(filename, 'r') as file:
    for line in file:
        topicList.append(line.strip())

functions = [
    {
        "name": "categorizeQuestion",
        "description": "Categorizes a question based on the provided topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic under which the question falls.",
                    "enum": topicList
                }
            },
            "required": ["topic"]
        }
    },
]

messages = []
topics_string = ', '.join(topicList)
message_content = "You are a useful assistant that categorises questions into one of these topics ({}). Each question may involve than one topic.".format(topics_string)
messages.append({"role": "system", "content": message_content})

#TODO parse questions individually
# reader = PdfReader("2023 Spring Midterm 1.pdf") 
# number_of_pages = len(reader.pages)
# page = reader.pages[14]
# question = page.extract_text()

def runCategorisellm(question):
    messages.append({"role": "user", "content": question}) 
    chatResponse = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        functions= functions,
        function_call={"name": "categorizeQuestion"}
    )
    topic= json.loads(chatResponse["choices"][0]["message"]["function_call"]["arguments"]).get("topic")
    return(topic)
