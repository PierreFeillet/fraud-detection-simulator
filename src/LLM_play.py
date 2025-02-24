import ollama
from datetime import datetime, timedelta
import json
from IPython import embed
import pandas as pd
import re
import numpy as np


def generate_agent_strategy(is_fraud: bool, filename="agent_strategy_response.txt"):
    """Queries the LLM to generate a strategy for a fraudster or legitimate user and saves the response."""
    prompt = (
        "You're in a simulation of " + ("fraudulent behavior" if is_fraud else "normal financial behavior") + " in banking. "
        "You are a " + ("clever fraudster" if is_fraud else "legitimate bank customer") + ". "
        "Describe your financial behavior and approach in detail but limited to what in practice a bank system would see. Your output strategy will be used to feed another LLM that must generates activities consistent with your startegy "
        "Specifically, the next LLM will have to generate the sequence of activities to achieve teh goal you defined in your strategy. Since we are in a simulate banking system, we only care about the activites *relevant* to a bank (not if for example the fraudster goes to parties with bank operators)."
        "The next agent will have to create a JSON file with the following information provided so be clear about the strategy for picking the right choices:"
        f"- Example format:\n"
        f"```json\n"
        f"[\n"
        f"    {{\"type\": \"Deposit\", \"amount\": 5000, \"location\": \"USA\", \"timestamp\": \"2025-03-05 09:00:00\", \"granted\": true}},\n"
        f"    {{\"type\": \"Transfer\", \"amount\": 2000, \"location\": \"Cayman Islands\", \"timestamp\": \"2025-03-06 11:30:00\", \"granted\": true}}\n"
        f"]\n" 
        "Provide a clear thought process, but do NOT return JSON."
    )

    # Send request to the LLM
    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
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
        f"- Each activity must include `type`, `amount`, `location`, `timestamp`.\n"
        f"- **Ensure `type` is concise** (e.g., 'Deposit', 'Transfer', 'Withdraw', 'Purchase').\n"
        f"- The **amount must be 0** if the activity does not involve a financial transaction.\n"
        f"- The **location must be aligned** with the agent's behavior (e.g., fraudsters use offshore locations, travelers move frequently).\n"
        f"- **If a transaction is not possible due to insufficient funds, mark `granted: false` and set `amount: 0`**.\n"
        f"- Return JSON format **only**, with no explanations.\n"
        f"- Example format:\n"
        f"```json\n"
        f"[\n"
        f"    {{\"type\": \"Deposit\", \"amount\": 5000, \"location\": \"USA\", \"timestamp\": \"2025-03-05 09:00:00\", \"granted\": true}},\n"
        f"    {{\"type\": \"Transfer\", \"amount\": 2000, \"location\": \"Cayman Islands\", \"timestamp\": \"2025-03-06 11:30:00\", \"granted\": true}}\n"
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

    #  Convert activity names & track balance with overdraft protection
    balance = initial_balance
    for activity in activity_sequence:
        #activity["type"] = activity_mapping.get(activity["type"], activity["type"])  # Shorten names
        activity["granted"] = True  # Default to granted

        if activity["amount"] > 0:
            if activity["type"] == "Deposit":
                balance += activity["amount"]  #  Increase balance for deposits
            else:
                #  Overdraft protection: Reject if insufficient funds
                if balance >= activity["amount"]:
                    balance -= activity["amount"]  #  Deduct balance normally
                else:
                    activity["granted"] = False  #  Mark as rejected
                    activity["amount"] = 0  # Reset amount since the transaction failed

        activity["balance"] = balance  # Track running balance

    return activity_sequence

def store_activities(activity_sequence):
    """Stores generated activities into a pandas DataFrame."""
    df = pd.DataFrame(activity_sequence)
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Ensure timestamps are datetime format
    return df


def evaluate_simulation(activity_sequence):
    """Computes a fraud probability score based on time gaps, amount variance, location changes, and transaction failures."""

    if len(activity_sequence) < 2:
        return {"score": 0.5, "error": "Not enough data for evaluation"}  # Neutral score if too few transactions

    df = pd.DataFrame(activity_sequence)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    #  **Compute Time Gap Score**
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    avg_time_gap = df["time_diff"].mean()
    time_score = min(1, avg_time_gap / 86400)  # Normalize (1 day gap = 1, <1 day = closer to 0)

    #  **Compute Amount Variation Score**
    amount_variance = np.var(df["amount"])
    amount_score = min(1, 1 / (1 + amount_variance / 10000))  # Normalize variance (high variance → fraud)

    #  **Compute Location Switching Score**
    unique_locations = df["location"].nunique()
    location_score = min(1, 1 / (1 + unique_locations / 5))  # More unique locations → Lower score

    #  **Compute Denied Transaction Score**
    denied_ratio = df[df["granted"] == False].shape[0] / df.shape[0]
    denied_score = 1 - min(1, denied_ratio)  # More denied transactions → Lower score

    #  **Final Score (Weighted)**
    final_score = 0.3 * time_score + 0.3 * amount_score + 0.2 * location_score + 0.2 * denied_score

    return {
        "Time Gap Score": round(time_score, 2),
        "Amount Stability Score": round(amount_score, 2),
        "Location Stability Score": round(location_score, 2),
        "Denied Transactions Score": round(denied_score, 2),
        "Final Fraud Score": round(final_score, 2)
    }


# Step 1: Generate an Agent Strategy
is_fraud = True  # Change to False for legitimate user
strategy = generate_agent_strategy(is_fraud)

if strategy:
    print(f"\nGenerated Strategy:\n{strategy}\n")

    # Step 2: Generate an Activity Sequence
    activities = generate_activity_sequence(strategy)

    if activities:
        # Step 3: Store in DataFrame
        df = store_activities(activities)
        print("\nGenerated Activity Sequence:\n")
        print(df)
        df.to_csv(f"generated_activities.csv", index=True)

        # Step 4: Evaluate Fraud Score
        scores = evaluate_simulation(activities)
        print("\nFraud Evaluation Metrics:")
        print(scores)
