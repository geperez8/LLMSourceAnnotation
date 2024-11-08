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
    

error = ""

st.write(f"{error}")

if st.button("Generate Text"):

    if context == "" or src_text == "":
        error = "Please enter a headline and source text to generate text."
    else:
        error = ''
        response = run_ollama(prompt, src_text)
        st.write(f"{response}")

    