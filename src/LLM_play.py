import ollama
from datetime import datetime, timedelta
import json
from IPython import embed
import pandas as pd
import re

activity_mapping = {
    "Deposit": "Deposit",
    "Transfer": "Transfer",
    "Withdrawal": "Withdraw",
    "Purchase": "Purchase",
    "Account Opening": "Open Account",
    "Login": "Login",
    "Phishing Attempt": "Phishing",
    "Offshore Transfer": "Transfer",
    "Luxury Spending": "Purchase"
}


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

    #  Save the response to a text file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(strategy)

    return strategy

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

def generate_activity_sequence(strategy: str, initial_balance=10000):
    """Queries the LLM to generate a structured sequence of activities based on a strategy, while tracking balance."""
    
    prompt = (
        f"You are an AI that generates structured financial activity sequences based on the following strategy:\n\n"
        f"### Strategy:\n{strategy}\n\n"
        f"### Instructions:\n"
        f"- Generate at least **5 activities** that align with the strategy.\n"
        f"- Each activity must include `type`, `amount`, `location`, and `timestamp`.\n"
        f"- The **amount must be 0** if the activity does not involve a financial transaction.\n"
        f"- The **location must be aligned** with the agent's behavior (e.g., fraudsters use offshore locations, travelers move frequently).\n"
        f"- Return JSON format **only** with no explanations.\n"
        f"- Example format:\n"
        f"```json\n"
        f"[\n"
        f"    {{\"type\": \"Deposit\", \"amount\": 5000, \"location\": \"USA\", \"timestamp\": \"2025-03-05 09:00:00\"}},\n"
        f"    {{\"type\": \"Transfer\", \"amount\": 2000, \"location\": \"Cayman Islands\", \"timestamp\": \"2025-03-06 11:30:00\"}}\n"
        f"]\n"
        f"```"
    )

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    raw_response = response['message']['content'].strip()

    #  Extract JSON safely
    activity_sequence = extract_json(raw_response)
    
    if not activity_sequence:
        print("Error: Failed to generate a valid activity sequence.")
        return None

    #  Convert activity names & track balance
    balance = initial_balance
    for activity in activity_sequence:
        activity["type"] = activity_mapping.get(activity["type"], activity["type"])  # Shorten names

        #  Balance update logic
        if activity["amount"] > 0:
            if activity["type"] == "Deposit":
                balance += activity["amount"]  # Increase balance for deposits
            else:
                balance -= activity["amount"]  # Decrease balance for withdrawals, transfers, purchases

        activity["balance"] = balance  # Track running balance

    return activity_sequence


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

