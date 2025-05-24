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

## Prompt
"Write a Python program to read a CSV file, calculate the average of a numeric column named 'Score', and print the result."

### 1. ChatGPT (OpenAI GPT-4)
````
import csv

def calculate_average_score(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        scores = [float(row['Score']) for row in reader if row['Score']]
    average = sum(scores) / len(scores)
    print(f"Average Score: {average:.2f}")

# Example usage
calculate_average_score('data.csv')
````

## 2. Gemini (Google AI)
````
import pandas as pd

def average_score(filepath):
    df = pd.read_csv(filepath)
    average = df['Score'].mean()
    print(f"Average Score: {average:.2f}")

# Example usage
average_score("data.csv")
````

## 3.Copilot 
````
def average_score_from_csv(file_path):
    total = 0
    count = 0
    with open(file_path, 'r') as file:
        next(file)  # skip header
        for line in file:
            score = float(line.strip().split(',')[-1])
            total += score
            count += 1
    print("Average Score:", total / count)

# Example
average_score_from_csv("data.csv")
````

##  Comparison Table

| Feature          | ChatGPT (GPT-4)          | Gemini (Google)           | Copilot (Microsoft)              |
| ---------------- | ------------------------ | ------------------------- | -------------------------------- |
| **Approach**     | `csv.DictReader`         | `pandas.read_csv()`       | Manual line + split parsing      |
| **Readability**  |  Clear & standard       |  Very readable & concise |  Prone to CSV format issues    |
| **Correctness**  |  High                   |  High                    |  Risky if CSV has extra fields |
| **Dependencies** |  None (Standard lib)    | Needs `pandas`         |  None                           |
| **Robustness**   |  With `if row['Score']` | NaN-safe via `mean()`   |  Assumes last column = Score    |
| **Ease of Use**  | Simple                 | If pandas installed     | Very minimal                   |

## Insight
### ChatGPT:
 gives the most balanced and robust solution, suitable for real-world usage.
### Gemini:
 provides the most concise solution (if you're okay with external libraries).
### Copilot:
 gives quick and minimal code, but lacks data validation.


# comclusion
This Python-based automation system successfully integrates multiple AI APIs to standardize input prompts, retrieve model outputs, compare them semantically, and produce actionable insights. The solution can be extended further to support bulk prompts, visualization dashboards, or automated reporting pipelines. It provides a valuable framework for benchmarking AI tools, conducting experiments, or even choosing the most suitable LLM for specific tasks in production environments.


