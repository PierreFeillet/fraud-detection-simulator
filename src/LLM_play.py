import ollama
from datetime import datetime, timedelta
import json
from IPython import embed
import pandas as pd

def generate_agent_strategy(is_fraudster: bool, filename="agent_strategy_response.txt"):
    """Queries the LLM to generate a strategy for a fraudster or legitimate user and saves the response."""
    prompt = (
        "You're in a simulation of " + ("fraudulent behavior" if is_fraudster else "normal financial behavior") + " in banking. "
        "You are a " + ("clever fraudster" if is_fraudster else "legitimate bank customer") + ". "
        "Describe your financial behavior and approach in detail.\n\n"
        "Your response should be a well-structured explanation of your strategy, "
        "such as how you spend money, transfer funds, or conduct fraud. "
        "Provide a clear thought process, but do NOT return JSON."
    )

    # Send request to the LLM
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    strategy = response['message']['content'].strip()

    # ✅ Save the response to a text file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(strategy)

    return strategy


import re
import json

def extract_json(text):
    """Extracts JSON content from a text response using regex, with fallback handling."""
    
    # First Attempt: Direct JSON Parsing (Best case scenario)
    try:
        return json.loads(text.strip())  # If response is pure JSON, this works
    except json.JSONDecodeError:
        pass  # Fall back to regex extraction

    # Second Attempt: Extract JSON Block (If LLM adds explanations)
    match = re.search(r"\[.*\]", text, re.DOTALL)  # Look for a JSON array
    if match:
        json_data = match.group(0)  # Extract JSON content
        try:
            return json.loads(json_data)  # Convert to Python list
        except json.JSONDecodeError:
            print("Error: Extracted JSON is invalid.")
            return None

    # If all else fails
    print("Error: No valid JSON found in LLM response.")
    return None


def generate_activity_sequence(strategy: str):
    """Queries the LLM to generate a structured sequence of activities based on a strategy."""
    
    prompt = (
        f"You are an AI that generates structured financial activity sequences based on the following strategy:\n\n"
        f"### Fraud/Legitimate Strategy:\n{strategy}\n\n"
        f"### Instructions:\n"
        f"- Generate at least **5 activities** that align with the strategy.\n"
        f"- Ensure timestamps are **realistic and sequential**.\n"
        f"- If it's a **fraudster**, create a pattern of fraudulent transactions.\n"
        f"- If it's a **legitimate user**, create typical everyday transactions.\n"
        f"- **Return only valid JSON**, with no explanations, no formatting, and no markdown.\n"
        f"- Your response **must start with `[` and end with `]`** (a valid JSON array).\n"
    )

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    raw_response = response['message']['content'].strip()

    # Extract JSON safely
    activity_sequence = extract_json(raw_response)

    if activity_sequence:
        return activity_sequence
    else:
        print("Error: Failed to generate a valid activity sequence.")
        return None

def store_activities(activity_sequence):
    """Stores generated activities into a pandas DataFrame."""
    df = pd.DataFrame(activity_sequence)
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Ensure timestamps are datetime format
    return df

# Step 1: Generate an Agent Strategy
is_fraudster = True  # Change to False for legitimate user
strategy = generate_agent_strategy(is_fraudster)

if strategy:
    print(f"\nGenerated Strategy:\n{strategy}\n")

    # Step 2: Generate an Activity Sequence
    activities = generate_activity_sequence(strategy)

    if activities:
        # Step 3: Store in DataFrame
        df = store_activities(activities)
        print("\nGenerated Activity Sequence:")
        print(df)
        df.to_csv(f"generated_activities.csv", index=True)

