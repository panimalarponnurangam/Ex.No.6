# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

### Date:
### Register no:212222110031
### NAME: PANIMALAR P
# Aim: 
Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# Algorithm
- Define a prompt or task to test (e.g., sentiment analysis or summarization).
- Send the prompt to different AI APIs.
- Collect and normalize responses.
- Compare outputs using similarity metrics (e.g., cosine similarity).
- Summarize differences and generate insights or recommendations.

# Objective
- Automate the interaction with multiple AI APIs.
- Benchmark outputs for consistency and quality.
- Generate comparative insights to assist in tool selection and use-case alignment.

# Reporting
- Output from each tool
- Similarity scores
- Recommended tool based on accuracy or context suitability
- Log file for audits

 # Required Libraries
 ````
pip install openai transformers cohere scikit-learn pandas python-dotenv
````
# Python Code
````
import openai
import cohere
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Step 1: Define input
task_prompt = "Summarize the importance of cybersecurity in IoT systems."

# Step 2: Get OpenAI response
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Step 3: Get Cohere response
def get_cohere_response(prompt):
    response = co.generate(model="command", prompt=prompt, max_tokens=100)
    return response.generations[0].text.strip()

# Step 4: Get Hugging Face response (using summarization pipeline)
def get_huggingface_response(prompt):
    summarizer = pipeline("summarization")
    return summarizer(prompt, max_length=60, min_length=30, do_sample=False)[0]['summary_text']

# Step 5: Compare outputs
def compare_responses(responses):
    vectorizer = TfidfVectorizer().fit_transform(responses)
    similarity = cosine_similarity(vectorizer)
    return similarity

# Execution
responses = {
    "OpenAI": get_openai_response(task_prompt),
    "Cohere": get_cohere_response(task_prompt),
    "HuggingFace": get_huggingface_response(task_prompt)
}

# Similarity matrix
similarity_matrix = compare_responses(list(responses.values()))

# Reporting
df = pd.DataFrame(responses.items(), columns=["Tool", "Response"])
print(df)
print("\nSimilarity Matrix:")
print(pd.DataFrame(similarity_matrix, index=responses.keys(), columns=responses.keys()))
````
# Output
![image](https://github.com/user-attachments/assets/15b2b538-c279-41ed-b438-71c374501aac)

# Result
This Python-based automation system successfully integrates multiple AI APIs to standardize input prompts, retrieve model outputs, compare them semantically, and produce actionable insights. The solution can be extended further to support bulk prompts, visualization dashboards, or automated reporting pipelines. It provides a valuable framework for benchmarking AI tools, conducting experiments, or even choosing the most suitable LLM for specific tasks in production environments.


