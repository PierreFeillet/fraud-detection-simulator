#  Top 10 Banking Frauds
TOP_10_FRAUD_TYPES = [
    "Money Laundering", "Account Takeover", "Synthetic Identity Fraud",
    "Card Skimming", "Loan Fraud", "Check Fraud", 
    "Wire Fraud", "Ponzi Scheme", "Cryptocurrency Fraud", "Insider Trading"
]

#  Top 10 Legitimate Banking Profiles
TOP_10_LEGITIMATE_PROFILES = [
    "Saver", "Investor", "Traveler", "Everyday Spender", 
    "Business Owner", "Student", "Retiree", "Frequent Online Shopper",
    "Tech Professional", "Freelancer"
]

FIXED_SCHEMA = [
    "transaction_id", "timestamp", "type", "amount", "currency", "account_id", "user_id",
    "balance_before", "balance_after", "location", "ip_address", "device_id", "network_type",
    "merchant_name", "recipient_id", "recipient_bank", "granted", "is_suspicious", "fraud_score"
]

import random
import json
import ollama
import os
import re
from IPython import embed

import re
import json

import re
import json

def extract_json(text):
    """Extracts the JSON block from a text response using custom markers JSON-START and JSON-STOP."""
    
    # Search for JSON content between markers
    match = re.search(r'JSON-START\n(.*?)\nJSON-STOP', text, re.DOTALL)
    print(match)
    
    if not match:
        print("⚠️ No valid JSON found in response.")
        return None

    json_text = match.group(1).strip()  # Extract matched JSON
    try:
        data = json.loads(json_text)  # Parse JSON string into a Python object
        return data if isinstance(data, list) else [data]  # Ensure it returns a list
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None



def generate_fraud_strategy(fraud_type=None, filename="strategies/fraud_strategy.json"):
    """Generates a fraud strategy based on a specific fraud type (or picks one randomly)."""
    os.makedirs('strategies', exist_ok=True)
    #  Pick a fraud type if none is provided
    if fraud_type is None:
        fraud_type = random.choice(TOP_10_FRAUD_TYPES)

    fraud_description = (
        f"You are a clever fraudster specializing in {fraud_type}. "
        f"Your goal is to execute a fraud strategy that banks might detect but in a way that minimizes your risk. "
        f"Describe your method realistically, limited to what a banking system can observe. "
        f"Outline the step-by-step approach, key financial activities, and tactics to avoid detection."
    )

    prompt = (
        "You're in a banking simulation. Your goal is to execute a fraudulent scheme. "
        f"{fraud_description} "
        "Your strategy will be used by another LLM to generate a sequence of financial activities aligned with your plan. "
        "Provide:\n"
        "- The goal of your fraud scheme\n"
        "- The specific fraud type (e.g., money laundering, account takeover, synthetic identity fraud, card skimming)\n"
        "- The key financial tactics used (e.g., multiple small transactions, shell companies, offshore transfers)\n"
        "- A structured plan outlining the activities needed to reach your goal\n"
        "Provide a clear thought process, but do NOT return JSON."
    )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()

    strategy_data = {
        "fraud_type": fraud_type,
        "fraud_strategy": strategy_text
    }

    # Save to JSON
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(strategy_data, file, indent=4)
    
    return strategy_data, strategy_text


def generate_legitimate_strategy(profile_type=None, filename="strategies/legitimate_strategy.json"):
    """Generates a legitimate banking strategy based on a customer profile type."""
    os.makedirs('strategies', exist_ok=True)

    #  Pick a profile type if none is provided
    if profile_type is None:
        profile_type = random.choice(TOP_10_LEGITIMATE_PROFILES)

    profile_description = (
        f"You are a legitimate bank customer with a {profile_type} profile. "
        f"Describe your typical financial behavior, transactions, and approach to money management."
    )

    prompt = (
        "You're in a banking simulation. You are a legitimate customer. "
        f"{profile_description} "
        "Your strategy will be used by another LLM to generate a sequence of financial activities aligned with your plan. "
        "Provide:\n"
        "- The financial goal of your behavior\n"
        "- Your customer profile (e.g., Saver, Investor, Traveler, Everyday Spender)\n"
        "- The key financial habits (e.g., monthly savings, frequent small purchases, international spending)\n"
        "- A structured plan outlining the expected activities\n"
        "Provide a clear thought process, but do NOT return JSON."
    )

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()

    strategy_data = {
        "profile_type": profile_type,
        "legitimate_strategy": strategy_text
    }

    # Save to JSON
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(strategy_data, file, indent=4)

    return strategy_data, strategy_text


def generate_activity_sequence(strategy: str, initial_balance=10000, currency="USD",
                               log_text_file="activity_log.txt", log_json_file="activity_log.json"):
    """Generates structured financial activities, ensuring all fields are present, and saves reasoning and JSON logs."""

    prompt = (
        f"You are an AI generating a **detailed timeline** of banking activities based on the strategy below.\n\n"
        f"### Strategy:\n{strategy}\n\n"
        f"### Instructions:\n"
        f"- First, explain your reasoning step by step.\n"
        f"- Then, generate only the structured JSON activity sequence.\n"
        f"- The JSON must be enclosed within JSON-START and JSON-STOP. Don't generate anything which is not JSON in the middle.\n"
        f"- Example format:\n"
        f"Reasoning:\n"
        f"--- START REASONING ---\n"
        f"Here’s how I will create the sequence...\n"
        f"--- END REASONING ---\n\n"
        f"JSON-START\n"
        f"[\n"
        f"    {{\"transaction_id\": \"TXN00001\", \"timestamp\": \"2025-03-01 12:00:00\", \"type\": \"Login\", \"amount\": 0, \"currency\": \"{currency}\", \"account_id\": null, \"user_id\": \"USER789\", \"balance_before\": null, \"balance_after\": null, \"location\": \"New York, USA\", \"ip_address\": \"192.168.1.10\", \"device_id\": \"iPhone-14\", \"network_type\": \"Wi-Fi\", \"merchant_name\": null, \"recipient_id\": null, \"recipient_bank\": null, \"granted\": null, \"is_suspicious\": false, \"fraud_score\": 0.0}}\n"
        f"]\n"
        f"JSON-STOP\n"
    )

    # ✅ Send the request to the LLM
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    raw_response = response['message']['content'].strip()

    # ✅ Extract Reasoning
    reasoning_match = re.search(r'--- START REASONING ---\n(.*?)\n--- END REASONING ---', raw_response, re.DOTALL)
    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "⚠️ No explicit reasoning found."
    
    save_to_text(log_text_file, raw_response)

    # ✅ Extract JSON Sequence
    activity_sequence = extract_json(raw_response)

    if not activity_sequence:
        print("❌ Error: No valid activity sequence extracted.")
        return None

    # ✅ Save reasoning & JSON to files
    #save_to_text(log_text_file, reasoning_text, activity_sequence)
    #save_to_json(log_json_file, reasoning_text, activity_sequence)

    return activity_sequence, reasoning_text


import pandas as pd
def activities_to_dataframe(activities, label):
    """Converts an activity sequence to a pandas DataFrame and adds a label for fraud/legit."""
    df = pd.DataFrame(activities)
    df["label"] = label  # Add a column to distinguish fraud vs legitimate
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Ensure timestamps are in datetime format
    return df

def save_to_json(json_filename, reasoning_text, json_data):
    """Appends LLM reasoning and extracted JSON as structured JSON objects."""
    
    log_entry = {
        "reasoning": reasoning_text,
        "activities": json_data
    }

    # Load existing data if file exists
    try:
        with open(json_filename, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Append new log entry
    existing_data.append(log_entry)

    # Save back to file
    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4)

#def save_to_text(log_filename, reasoning_text, json_data):
#    """Appends LLM reasoning and extracted JSON to a shared text file."""
#    with open(log_filename, "a", encoding="utf-8") as log_file:
#        log_file.write(f"\n### LLM Chain of Thought ###\n\n{reasoning_text}\n\n")
#        log_file.write(f"### Extracted JSON Sequence ###\n\n{json.dumps(json_data, indent=4)}\n")
#        log_file.write("\n" + "=" * 80 + "\n")  # Separator for readability

def save_to_text(log_filename, reasoning_text):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought ###\n\n{reasoning_text}\n\n")



# Step 1: Generate Fraud & Legitimate Strategies
fraud_strategy_data, fraud_strategy_text = generate_fraud_strategy()
legit_strategy_data, legit_strategy_text = generate_legitimate_strategy()


# Step 2: Generate Activity Sequences
fraud_activities, fraud_reasoning = generate_activity_sequence(fraud_strategy_text, initial_balance=5000)
legit_activities, legit_reasoning = generate_activity_sequence(legit_strategy_text, initial_balance=10000)

# Step 3: Convert Activities to DataFrames
fraud_df = activities_to_dataframe(fraud_activities, label="fraud")
legit_df = activities_to_dataframe(legit_activities, label="legitimate")

# Step 4: Combine Both DataFrames and Save to CSV
full_df = pd.concat([fraud_df, legit_df]).sort_values(by="timestamp").reset_index(drop=True)
csv_filename = "banking_activity_log.csv"
full_df.to_csv(csv_filename, index=False)

# Step 5: Display the first few rows
import ace_tools as tools
tools.display_dataframe_to_user(name="Banking Activity Log", dataframe=full_df)

print(f"\n Banking activity log saved to: {csv_filename}")
