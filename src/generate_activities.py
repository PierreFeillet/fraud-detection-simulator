from datetime import datetime
import json
import os
import random
import re
import uuid
from IPython import embed
import pandas as pd
import ollama


def generate_activity_sequence(strategy, currency="USD", fraud_label=0, profile_type="Legitimate", global_clock=None, user_id=None, log_text_file="outputs/llm_chain_of_thought.txt"):
    """Generates structured financial activities based on the provided strategy."""
    prompt = (
        f"You are an AI generating a **detailed timeline** of banking activities.\n"
        f"You are in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions (like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time.\n"
        f"You must generate activities in JSON format based on the strategy below.\n\n"
        f"### Strategy:\n{strategy}\n\n"
        f"### Instructions:\n"
        f"- First, explain your reasoning step by step without making examples.\n"
        f"- Then, generate **only** the structured JSON activity sequence.\n"
        f"- The JSON MUST be enclosed within **triple backticks** using the format ```json ... ```.\n"
        f"- Do **not** add any text after the JSON block.\n"
        f"- **Do NOT repeat the example JSON in your reasoning.**\n"
        f"- **If a transaction is not possible due to insufficient funds, mark `granted: false` and set `amount: 0`.**\n"
        f"- `balance_before` must be initialized with a realistic value if it's the first time generating the activity sequence for that `account_id`.\n"
        f"- `balance_after` must be updated according to the transaction type and the amount of the transaction.\n"
        f"- Each activity must have two timestamps: `bank_timestamp` and `local_timestamp`.\n"
        f"    - `bank_timestamp` represents the time in the bank's time zone (UTC) and must be formatted as 'YYYY-MM-DDTHH:MM:SS+00:00'.\n"
        f"    - `local_timestamp` represents the time in the agent's local time zone and must include the local offset (e.g., 'YYYY-MM-DDTHH:MM:SS-05:00').\n"
        f"- The `velocity` field must represent the time delta in minutes between the current activity's `bank_timestamp` and the previous activity's `bank_timestamp` in minutes. If it's the first transaction is a realistic random number.\n"
        f"- The `distance_from_last_location` field must represent the distance in kilometers from the previous activity's `location` to the current activity's `location`. If it's the first transaction is a realistic random number.\n"
        f"- Ensure that the timestamps are logically consistent and formatted according to ISO 8601 standards.\n"
        f"- The JSON output MUST follow this structure and HAVE the following fields:\n"
        f"```json\n"
        f"[\n"
        f"    {{\n"
        f"        \"transaction_id\":  \"{uuid.uuid4()}\",\n"
        f"        \"bank_timestamp\": \"{global_clock}+00:00\",\n"
        f"        \"local_timestamp\": \"2025-03-01T07:00:00-05:00\",\n"
        f"        \"user_id\": \"{user_id}\",\n"
        f"        \"account_id\": ACC12345,\n"
        f"        \"type\": \"Purchase\",\n"
        f"        \"amount\": 150.75,\n"
        f"        \"currency\": \"{currency}\",\n"
        f"        \"balance_before\": 1000,\n"
        f"        \"balance_after\": 849.25,\n"
        f"        \"location\": \"New York, USA\",\n"
        f"        \"ip_address\": \"192.168.1.10\",\n"
        f"        \"device_id\": \"iPhone-14\",\n"
        f"        \"network_type\": \"Wi-Fi\",\n"
        f"        \"merchant_name\": Amazon,\n"
        f"        \"recipient_id\": null,\n"
        f"        \"recipient_bank\": null,\n"
        f"        \"granted\": True,\n"
        f"        \"login_attempts\": 1,\n"
        f"        \"session_id\": \"SESSION123\",\n"
        f"        \"velocity\": 15,\n"
        f"        \"distance_from_last_location\": 20,\n"
        f"        \"is_repeat_location\": true,\n"
        f"        \"fraud_label\": {fraud_label},\n"
        f"        \"behavior_type\": \"{profile_type}\"\n"
        f"    }}\n"
        f"]\n"
        f"```\n"
        f"\n"
        f"### Explanation of Each Field:\n"
        f"- `transaction_id`: Unique identifier for the transaction generated using UUID.\n"
        f"- `user_id`: Unique identifier for the user generated using UUID.\n"
        f"- `bank_timestamp`: The date and time when the activity occurred in UTC, formatted as 'YYYY-MM-DDTHH:MM:SS+00:00'.\n"
        f"- `local_timestamp`: The date and time when the activity occurred in the local time zone, including the time zone offset.\n"
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
        f"- `login_attempts`: Number of login attempts in the session.\n"
        f"- `session_id`: Unique identifier for the session grouping multiple activities.\n"
        f"- `velocity`: Time difference from the previous transaction to detect rapid actions.\n"
        f"- `distance_from_last_location`: Distance in km from the previous activity location to detect impossible travel.\n"
        f"- `is_repeat_location`: Boolean flag indicating if the transaction is from a familiar location.\n"
        f"- `fraud_label`: Ground truth label for supervised learning (`1` for fraud, `0` for legitimate).\n"
        f"- `behavior_type`: Indicates the behavioral profile the activity sequence belongs to (e.g., Identity Theft, High Frequency Traveler, Student, Card Skimming).\n"
        f"End of your task.\n"
    )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    raw_response = response['message']['content'].strip()
    save_to_text(log_text_file, raw_response)

    activity_sequence = extract_json(raw_response)
    return activity_sequence

def save_to_text(log_filename, reasoning_text):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought ###\n\n{reasoning_text}\n\n")

def extract_json(text):
    """Extracts all JSON arrays from the LLM response, handling both single and multiple blocks."""
    
    # Find all JSON arrays enclosed within triple backticks
    matches = re.findall(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)

    if not matches:
        # Fallback: find JSON arrays without triple backticks
        matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)

    if not matches:
        print("⚠️ No valid JSON arrays found in the response.")
        return None

    combined_activities = []

    for json_text in matches:
        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                combined_activities.extend(data)
            else:
                combined_activities.append(data)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON block: {e}")
            continue

    return combined_activities if combined_activities else None

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20):
    """Generates a bank log with multiple fraudulent and legitimate agents, ensuring a realistic fraud ratio."""
    
    all_activities = []
    fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
    legitimate_strategies = load_existing_strategies("strategies/legitimate_strategies.json")

    # Global clock for the start of the simulation
    global_clock = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%S')

    # Generate fraudulent activities for multiple agents
    fraud_activities_count = 0
    for _ in range(fraud_agents_count):
        user_id = str(uuid.uuid4())
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]
        fraud_label = 1

        activities = generate_activity_sequence(strategy=strategy, fraud_label=fraud_label, profile_type=behavior_type, global_clock=global_clock, user_id=user_id)
        for activity in activities:
            all_activities.append(activity)
            fraud_activities_count += 1
            if fraud_activities_count >= int(total_activities * target_fraud_percentage):
                break
        if fraud_activities_count >= int(total_activities * target_fraud_percentage):
            break

    # Generate legitimate activities for multiple agents
    legit_activities_count = 0
    while len(all_activities) < total_activities:
        for _ in range(legit_agents_count):
            user_id = str(uuid.uuid4())
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]
            fraud_label = 0

            activities = generate_activity_sequence(strategy=strategy, fraud_label=fraud_label, profile_type=behavior_type, global_clock=global_clock, user_id=user_id)
            for activity in activities:
                all_activities.append(activity)
                legit_activities_count += 1
                if len(all_activities) >= total_activities:
                    break
            if len(all_activities) >= total_activities:
                break

    # Save to DataFrame and CSV
    df = pd.DataFrame(all_activities)
    df.to_csv("outputs/bank_log.csv", index=False)
    
    return df


def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

generate_activities(total_activities=15, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)