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
    """Extracts JSON from triple backticks (` ```json `) in the LLM response."""
    
    # Match content between triple backticks and 'json' keyword
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    
    if not match:
        print("⚠️ No valid JSON found in response.")
        return None

    json_text = match.group(1).strip()  # Extract matched JSON
    try:
        data = json.loads(json_text)  # Parse JSON string into a Python object
        return data if isinstance(data, list) else [data]  # Ensure it returns a list
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
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
        f"You're in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions (like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time.\n"
        f"Your goal is to execute a fraudulent scheme.\n"
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

    save_strategy_to_txt(fraud_type, strategy_text, "fraud_strategies.txt")
    
    return strategy_text

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
        f"You're in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions (like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time"
        f"Sometimes legitimate customer operation are not granted because of the bank alerts."
        f"You are a legitimate customer. "
        f"{profile_description} "
        "Your strategy will be used by another LLM to generate a sequence of financial activities aligned with your plan. "
        "Provide:\n"
        "- The financial goal of your behavior\n"
        "- Your customer profile (e.g., Saver, Investor, Traveler, Everyday Spender)\n"
        "- The key financial habits (e.g., monthly savings, frequent small purchases, international spending)\n"
        "- A structured plan outlining the expected activities\n"
        "Provide a clear thought process, but do NOT return JSON."
    )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()

    save_strategy_to_txt(profile_type, strategy_text, "legitimate_strategies.txt")

    return strategy_text


def generate_activity_sequence(strategy: str, initial_balance=10000, currency="USD",
                               log_text_file="activity_log.txt", log_json_file="activity_log.json"):
    """Generates structured financial activities, ensuring all fields are present, and saves reasoning and JSON logs."""
    prompt = (
    f"You are an AI generating a **detailed timeline** of banking activities.\n"
    f"You are in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions (like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time.\n"
    f"Sometimes legitimate customer operations are not granted because of bank alerts.\n"
    f"You must generate activities based on the strategy below.\n\n"
    f"### Strategy:\n{strategy}\n\n"
    f"### Instructions:\n"
    f"- First, explain your reasoning step by step.\n"
    f"- Then, generate **only** the structured JSON activity sequence.\n"
    f"- The JSON must be enclosed within **triple backticks** using the format ```json ... ```.\n"
    f"- Do **not** add any text after the JSON block.\n"
    f"- **If a transaction is not possible due to insufficient funds, mark `granted: false` and set `amount: 0`.**\n"
    f"- The JSON output should follow this structure:\n"
    f"```json\n"
    f"[\n"
    f"    {{\"transaction_id\": \"TXN00001\", \"timestamp\": \"2025-03-01 12:00:00\", \"type\": \"Login\", \"amount\": 0, \"currency\": \"{currency}\", \"account_id\": null, \"user_id\": \"USER789\", \"balance_before\": null, \"balance_after\": null, \"location\": \"New York, USA\", \"ip_address\": \"192.168.1.10\", \"device_id\": \"iPhone-14\", \"network_type\": \"Wi-Fi\", \"merchant_name\": null, \"recipient_id\": null, \"recipient_bank\": null, \"granted\": null, \"is_suspicious\": false, \"login_attempts\": 1, \"session_id\": \"SESSION123\", \"velocity\": null, \"distance_from_last_location\": null, \"device_trust_score\": 85, \"ip_reputation_score\": 10, \"compromised_device\": false, \"compromised_network\": false, \"is_repeat_location\": true, \"transaction_risk_score\": 5, \"fraud_label\": 0, \"behavior_type\": \"Student\" }}\n"
    f"]\n"
    f"```\n"
    f"\n"
    f"### Explanation of Each Field:\n"
    f"- `transaction_id`: Unique identifier for the transaction.\n"
    f"- `timestamp`: The date and time when the activity occurred.\n"
    f"- `type`: The type of activity (e.g., Login, Withdrawal, Purchase, Transfer).\n"
    f"- `amount`: The amount of money involved in the transaction.\n"
    f"- `currency`: The currency of the transaction.\n"
    f"- `account_id`: Identifier of the bank account involved.\n"
    f"- `user_id`: Identifier for the user performing the transaction.\n"
    f"- `balance_before`: The account balance before the transaction.\n"
    f"- `balance_after`: The account balance after the transaction.\n"
    f"- `location`: Geographic location of the activity.\n"
    f"- `ip_address`: IP address used during the activity.\n"
    f"- `device_id`: Device identifier (e.g., phone or computer model).\n"
    f"- `network_type`: Type of network used (e.g., Wi-Fi, Mobile Data).\n"
    f"- `merchant_name`: Name of the merchant if applicable.\n"
    f"- `recipient_id`: Identifier of the recipient for transfers.\n"
    f"- `recipient_bank`: Bank of the recipient.\n"
    f"- `granted`: Indicates if the transaction was approved (`true`) or denied (`false`).\n"
    f"- `is_suspicious`: Boolean flag indicating if the transaction is suspicious.\n"
    f"- `login_attempts`: Number of login attempts in the session.\n"
    f"- `session_id`: Unique identifier for the session grouping multiple activities.\n"
    f"- `velocity`: Time difference from the previous transaction to detect rapid actions.\n"
    f"- `distance_from_last_location`: Distance from the previous activity location to detect impossible travel.\n"
    f"- `device_trust_score`: A score representing the trust level of the device based on its usage history.\n"
    f"- `ip_reputation_score`: Score indicating the risk associated with the IP address.\n"
    f"- `compromised_device`: Boolean flag indicating if the device is known to be compromised.\n"
    f"- `compromised_network`: Boolean flag for risky networks (e.g., public Wi-Fi).\n"
    f"- `is_repeat_location`: Boolean flag indicating if the transaction is from a familiar location.\n"
    f"- `transaction_risk_score`: Overall risk score based on transaction features.\n"
    f"- `fraud_label`: Ground truth label for supervised learning (`1` for fraud, `0` for legitimate).\n"
    f"- `behavior_type`: Indicates whether the activity is part of a specific legitimate behavioral profile (e.g high frequency traveler, student, ) or a specific fraud type (e.g., Identity Theft, Money Laundering).\n"
    )



    # ✅ Send the request to the LLM
    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    raw_response = response['message']['content'].strip()

    # ✅ Extract Reasoning
    #reasoning_match = re.search(r'--- START REASONING ---\n(.*?)\n--- END REASONING ---', raw_response, re.DOTALL)
    #reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "⚠️ No explicit reasoning found."
    
    save_to_text(log_text_file, raw_response)

    # ✅ Extract JSON Sequence
    activity_sequence = extract_json(raw_response)

    if not activity_sequence:
        print("❌ Error: No valid activity sequence extracted.")
        return None

    # ✅ Save to JSON 
    save_to_json('activities.json', activity_sequence)

    return activity_sequence

def save_strategy_to_txt(profile_type, strategy_text, filename):
    os.makedirs("strategies", exist_ok=True)
    with open(f"strategies/{filename}", "a") as file:
        file.write(f"Profile Type: {profile_type}\n")
        file.write("Strategy:\n")
        file.write(f"{strategy_text}\n")
        file.write("---\n")
        
import pandas as pd
def activities_to_dataframe(activities, label):
    """Converts an activity sequence to a pandas DataFrame and adds a label for fraud/legit."""
    df = pd.DataFrame(activities)
    df["label"] = label  # Add a column to distinguish fraud vs legitimate
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Ensure timestamps are in datetime format
    return df

def save_to_json(json_filename, json_data):
    """Appends LLM reasoning and extracted JSON as structured JSON objects."""
    
    log_entry = {
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
    with open(json_filename, "a", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4)
        

def save_to_text(log_filename, reasoning_text):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought ###\n\n{reasoning_text}\n\n")



# Step 1: Generate Fraud & Legitimate Strategies
fraud_strategy_text = generate_fraud_strategy()
legit_strategy_text = generate_legitimate_strategy()


# Step 2: Generate Activity Sequences
fraud_activities = generate_activity_sequence(fraud_strategy_text, initial_balance=5000)
legit_activities = generate_activity_sequence(legit_strategy_text, initial_balance=10000)

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
