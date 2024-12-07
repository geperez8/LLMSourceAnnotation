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

# Streamlit App
st.title("LLM Document Annotation Demo")

context = st.text_area("What is the header of your article?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

src_text = st.text_area("What is the text for your source?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

# # create Prompt
prompt = """Identify and extract the most newsworthy excerpts from a given document, providing 3-5 excerpts ranked in order of perceived newsworthiness. Each excerpt should be transcribed exactly as it appears in the source, without any modification.

Consider the relevance, significance, and impact of each segment when determining "newsworthiness." Focus on elements that are likely to attract media attention, have public interest, or convey important changes, events, or statements. 

# Steps

1. **Read Through the Full Document**: Gain an understanding of the document's context and main themes.
2. **Identify Potential Excerpts**: Look for statements that are significant, surprising, publicly relevant, or convey critical information.
3. **Select and Rank Excerpts**: Pick 3-5 excerpts that are the most noteworthy, assessing based on public interest or potential news impact. Rank them in order, from most to least newsworthy.
4. **Preserve the Original Wording**: Ensure the excerpts are copied exactly as written in the original document to maintain their integrity.

# Output Format

The output should be in JSON format with the following structure:

```json
{
  "excerpts": [
    {
      "rank": 1,
      "excerpt": "[Full text of the most newsworthy excerpt]"
    },
    {
      "rank": 2,
      "excerpt": "[Full text of the second most newsworthy excerpt]"
    },
    {
      "rank": 3,
      "excerpt": "[Full text of the third most newsworthy excerpt]"
    }
    // Additional excerpts (rank 4 and 5) are optional
  ]
}
```

# Examples

**Input Example:**
"A new scientific report has confirmed significant changes in global weather patterns. In addition, government officials have announced ambitious new climate policies, which are expected to reduce carbon emissions by 50% by 2030. Meanwhile, protests have erupted in several cities opposing recent fuel price hikes, with reports of multiple arrests."

**Output Example:**
```json
{
  "excerpts": [
    {
      "rank": 1,
      "excerpt": "Government officials have announced ambitious new climate policies, which are expected to reduce carbon emissions by 50% by 2030."
    },
    {
      "rank": 2,
      "excerpt": "Protests have erupted in several cities opposing recent fuel price hikes, with reports of multiple arrests."
    },
    {
      "rank": 3,
      "excerpt": "A new scientific report has confirmed significant changes in global weather patterns."
    }
  ]
}
```

# Notes

- Ensure that the excerpts are factually representative of the original text.
- If the document lacks clearly newsworthy excerpts, select those with the greatest potential public interest.
- Maintain impartiality in the ranking process, and base the decision solely on the impact value of each statement.
"""



def run_ollama(prompt, src_text):
    # Ollama Payload
    payload = {
        "model": "llama3.1:8b",
        "system": prompt,
        "prompt": src_text,
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
        quote = item["excerpt"]
        ranking = item["rank"]
        
        # Search for quote in source text
        match = re.search(re.escape(quote), src_text)
        if match:
            start, end = match.span()

            # Add text before the match (if any) as regular text
            if start > last_index:
                annotations.append(src_text[last_index:start])
            
            # Add matched quote as annotated text
            annotations.append((src_text[start:end], str(ranking)))

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

is_local = st.checkbox("Local LLM")

if st.button("Generate Text"):

    if context == "" or src_text == "":
        error = "Please enter a headline and source text to generate text."
    else:
        error = ''
        if is_local == True:
            response = run_ollama(prompt, src_text)
        else:
            response = run_gpt(prompt)

        # Extract the JSON content using regex
        json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        
        if json_match:
            json_content = json_match.group(1)
            try:
                # Parse the extracted JSON content
                response_json = json.loads(json_content)
                st.write("Extracted JSON:")
                st.json(response_json)

                # Display the annotated text with quotes
                annotate_text_with_quotes(src_text, response_json["excerpts"])
                
            except json.JSONDecodeError:
                st.write("Error: Could not parse JSON content.")
        else:
            st.write("Error: JSON content not found in the response.")

    