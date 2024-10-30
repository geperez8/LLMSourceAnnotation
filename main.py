import requests
import streamlit as st

OLLAMA_API_URL = "http://localhost:11434/api/generate"

st.title("LLM Document Annotation Demo")

context = st.text_area("What is the header of your article?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")



src_text = st.text_area("What is the text for your source?", value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")


# def get_user_input(text_prompt):
#     print(text_prompt + " Type 'END' on a new line to finish:")

#     user_input = []
#     while True:
#         line = input()
#         if line == 'END':
#             break
#         user_input.append(line)

#     multiline_input = "\n".join(user_input)

#     return multiline_input

# # get article context
# context = get_user_input("What is the headline for your article?")

# # get source text
# source = get_user_input("What is the text for your source?")

# # create Prompt
prompt = f'''
I am writing an article titled:

{context}

I am interested in using this document as a source in my article:

{src_text}

What parts of the document would go well with my article? Return it as just the quotes without any additional information. Also give me your reasoning behind it. Give it to me as a json with a quote and reason key and make sure you do at least 5 quotes with reasoning. Your purspose is to just return the json without any additional text, if the input is invalid, return a blank json and no additional comments.
'''



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
    

error = ""

st.write(f"{error}")

if st.button("Generate Text"):

    if context == "" or src_text == "":
        error = "Please enter a headline and source text to generate text."
    else:
        error = ''
        response = run_ollama(prompt)
        st.write(f"{response}")

    