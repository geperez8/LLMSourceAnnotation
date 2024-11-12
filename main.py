import os
import re
import json
import requests
import streamlit as st
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from annotated_text import annotated_text
from pydantic import BaseModel, ValidationError

load_dotenv()

# OpenAI API
# get the API key from the environment variables
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_KEY"),
)

# set OpenAI's API URL
OPENAI_API_URL = "https://api.openai.com/v1/engines/davinci/completions"

# PyDantic Classes for Expected Resopnse
# Define a Pydantic model for each quote and reasoning pair
class QuoteReasonPair(BaseModel):
    quote: str
    reason: str

# Define a model for the entire JSON response, containing multiple pairs
class GPTResponseModel(BaseModel):
    quotes: List[QuoteReasonPair]

# Ollama API
# set Ollama's API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define the expected schema for the Ollama response
ollama_schema = {
    "quotes": [
        {"quote": str, "reason": str}
    ]
}

is_local = False

# Streamlit App
st.title("LLM Document Annotation Demo")

context = st.text_area("What is the header of your article?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

src_text = st.text_area("What is the text for your source?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

# # create Prompt
prompt = f"""
I am writing an article titled:

{context}

I am interested in using this document as a source in my article:

{src_text}

What parts of the document would go well with my article? Return it as just the quotes without any additional information. 
Also give me your reasoning behind it. Give it to me as a JSON with a "quote" and "reason" key and make sure you do at least 5 quotes with reasoning. 
Your purpose is to just return the JSON without any additional text; if the input is invalid, return a blank JSON, no additional comments, and make 
the quotes one to one with the text. There shouldn't be any abbreviations in the quotes, shortening, or periods where they are not in the source text.

Return the response in JSON format as:
{{
    "quotes": [
        {{"quote": "<quote>", "reason": "<reason>"}},
        ...
    ]
}}
"""

# Functions
def run_ollama(prompt):
    # Ollama Payload
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False
    }

    print("\nGenerating text...")

    # send the request to Ollama's API
    response = requests.post(OLLAMA_API_URL, json=payload)

    # check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nGenerated text:")
        print(result["response"])
        return result["response"]
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)
        return f"\nError: {response.status_code}"
    


def run_gpt(prompt):
    chat_completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-2024-08-06",
        response_format=GPTResponseModel
    )

    # print("\nGenerated text:")
    # print("Completion is -----", chat_completion, "\n\n")
    # print("Choices are -----", chat_completion.choices, "\n\n")
    # print("Messages are -----", chat_completion.choices[0].message, "\n\n")
    # print("Content is -----", chat_completion.choices[0].message.content, "\n\n")

    event = chat_completion.choices[0].message.parsed
    content = chat_completion.choices[0].message.content

    print("Parsed is -----", event, "\n\n")
    
    # Convert response to a JSON object
    response_data = json.loads(content)

    print(response_data, "\n\n")
        
    # Validate the JSON structure using Pydantic
    validated_response = GPTResponseModel(**response_data)
        
    # Return the validated JSON as a dictionary
    return validated_response.model_dump()

def annotate_text_with_quotes(src_text, quotes):
    annotations = []
    last_index = 0

    # Loop through each quote to find and annotate it in the source text
    for item in quotes:
        quote = item["quote"]
        reason = item["reason"]
        
        # Search for quote in source text
        match = re.search(re.escape(quote), src_text)
        if match:
            start, end = match.span()

            # Add text before the match (if any) as regular text
            if start > last_index:
                annotations.append(src_text[last_index:start])
            
            # Add matched quote as annotated text
            annotations.append((src_text[start:end], "quote"))

            # Update last index to end of match
            last_index = end

    # Append remaining text after last match, if any
    if last_index < len(src_text):
        annotations.append(src_text[last_index:])
    
    print(annotations)
    # Display annotated text using Streamlit
    annotated_text(*annotations)

error = ""

st.write(f"{error}")

if st.button("Generate Text"):

    if context == "" or src_text == "":
        error = "Please enter a headline and source text to generate text."
    else:
        error = ''
        if is_local == True:
            response = run_ollama(prompt)
        else:
            response = run_gpt(prompt)
        # print(response, "\n\n")
        st.json(response)
        annotate_text_with_quotes(src_text, response["quotes"])