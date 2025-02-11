from datetime import datetime, timedelta, timezone
import json
import os
import random
import re
import uuid
from IPython import embed
import pandas as pd
import ollama

def build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id):
    """Builds the prompt for generating activity sequences with dynamic values and data schema."""
    # JSON Template with Dynamic Values
    json_template = f"""
    ```json
    [
      {{
        "transaction_id": "{uuid.uuid4()}",
        "bank_timestamp": "{global_clock}+00:00",
        "local_timestamp": "2025-03-01T07:00:00-05:00",
        "user_id": "{user_id}",
        "account_id": "ACC12345",
        "type": "Purchase",
        "amount": 150.75,
        "currency": "USD",
        "balance_before": 1000,
        "balance_after": 849.25,
        "location": "New York, USA",
        "ip_address": "192.168.1.10",
        "device_id": "iPhone-14",
        "network_type": "Wi-Fi",
        "merchant_name": "Amazon",
        "recipient_id": null,
        "recipient_bank": null,
        "granted": true,
        "login_attempts": 1,
        "session_id": "SESSION123",
        "velocity": 0.52,
        "distance_from_last_location": 3.4,
        "is_repeat_location": true,
        "fraud_label": {fraud_label},
        "behavior_type": "{profile_type}"
      }}
    ]
    ```
    """
    field_explanation = """"
        f"- `transaction_id`: string, A unique identifier for the transaction generated using UUID.\n"
        f"- `bank_timestamp`: string (ISO 8601 format), The UTC time when the activity occurred, formatted as 'YYYY-MM-DDTHH:MM:SS+00:00'.\n"
        f"- `local_timestamp`: string (ISO 8601 format), The local time of the activity with time zone offset (e.g., 'YYYY-MM-DDTHH:MM:SS-05:00').\n"
        f"- `user_id`: string, A unique identifier for the user generated using UUID. Given \n"
        f"- `account_id`: string, The identifier of the bank account involved in the transaction.\n"
        f"- `type`: string, The type of activity (e.g., Purchase, Withdrawal, Transfer).\n"
        f"- `amount`: flaot, The monetary amount involved in the transaction.\n"
        f"- `currency`: string, The currency of the transaction (e.g., 'USD'). Must be set according to the location of the activty\n"
        f"- `balance_before`: flaot, The account balance before the transaction.\n"
        f"- `balance_after`: flaot, The account balance after the transaction (adjusted based on the `amount`). If the generated transaction would make `balance_after`<0 then the field `granted` must be set to `false` and `balance_after`=`balance_before`\n"
        f"- `location`: string, The city and country where the transaction took place.\n"
        f"- `ip_address`: string, The IP address used during the activity.\n"
        f"- `device_id`: string, The device identifier (e.g., phone or computer model).\n"
        f"- `network_type`: string, Type of network used (e.g., Wi-Fi, Mobile Data).\n"
        f"- `merchant_name`: string or null, Name of the merchant (if applicable, e.g., 'Amazon' otherwise, set to `null`).\n"
        f"- `recipient_id`: string or null, The identifier of the recipient in case of transfers; otherwise, set to `null`.\n"
        f"- `recipient_bank`: string or null, The bank of the recipient in case of transfers; otherwise, set to `null`.\n"
        f"- `granted`: bool, Boolean indicating if the transaction was approved (`true`) or denied (`false`).\n"
        f"- `login_attempts`: integer, Number of login attempts in the session.\n"
        f"- `session_id`: string, Unique identifier for the session grouping multiple activities.\n"
        f"- `velocity`: float, Numeric value representing the time delta in minutes between the current and previous `bank_timestamp`. Use a random value if it's the first transaction.\n"
        f"- `distance_from_last_location`: float, Numeric value in kilometers indicating the distance from the previous activity's location. Use a random value if it's the first transaction.\n"
        f"- `is_repeat_location`: bool, Boolean indicating if the transaction is from a previously used location.\n"
        f"- `fraud_label`: integer, `1` for fraudulent transactions, `0` for legitimate ones.\n"
        f"- `behavior_type`: string, Describes the behavioral profile (e.g., 'Account Takeover', 'Legitimate').\n\n"
    """

    # Final Prompt with Strategy and JSON Example
    return (
        f"You are an AI generating **detailed sequences of banking activities** for a fraud simulation.\n"  
        f"Your task is to create a sequence of financial activities based on the predefined strategy below.\n\n"
        f"### Strategy:\n{strategy}\n\n"
        f"If `{fraud_label}`=1: "
        f"  1. Ensure the sequence of activities **completes the fraud** as described in the strategy by the minimum number of activties."
        f"  2. If the fraud is a **multi-event fraud**, generate at least **`minimum_activities`** (spceified in the {strategy}) transactions that follow the specified pattern."
        f"  3. For **single-event fraud**, ensure the transaction fully represents the fraudulent behavior."
        f"  4. The generated activities must clearly reflect the fraud type, scope, and behavior characteristics provided in the strategy."
        f"### Data Generation Rules:\n"
        f"The financial activities must be generated as a JSON object.\n"
        f"### Required Fields in the JSON, with format specifications, don't add extra-fields:\n{field_explanation}\n\n"
        f"- First, explain your reasoning step by step **without making examples**.\n"
        f"- Then, generate **only** the structured JSON activity sequence as in the following JSON example: \n{json_template}\n"
        f"- The JSON MUST be enclosed within **triple backticks** using the format ```json ... ```.\n"
        f"- Ensure that timestamps are logically consistent and formatted according to ISO 8601 standards.\n"
        f"- Do **NOT** include comments or explanations in the JSON.\n"
        f"- End your response immediately after closing the JSON block.\n"
        f"- **Use the provided `fraud_label`, `profile_type`, `global_clock`, `user_id`. They must remain exactly as given: `{fraud_label},{profile_type}, {global_clock},{user_id}`.**\n"
        f"### Important:\n"
        f"- Ensure that the JSON is correctly formatted, with the correct formats.\n"
        f"- **The JSON must be flat**\n"
        f"- End your response after closing triple backticks.\n"
        f"- Ensure all field values match the types and formats specified in the JSON field explanation. If the format is string, the string must be enclosed in "" or ''\n"
    )



def generate_activity_sequence(strategy, fraud_label=0, profile_type="Legitimate", global_clock=None, user_id=None, log_text_file="outputs/llm_chain_of_thought.txt"):
    """Generates structured financial activities based on the provided strategy."""
    while True:
        prompt = build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id)
        response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
        raw_response = response['message']['content'].strip()
        save_to_text(log_text_file, raw_response, user_id)

        activity_sequence = extract_json(raw_response)

        if activity_sequence != 'retry':
            print('✅ Activity sequence generated successfully for user ID:', user_id)
            return activity_sequence # Return only if valid JSON is extracted

        print("🔄 Retrying activity sequence generation due to invalid JSON...")

def save_to_text(log_filename, reasoning_text, agent_id=None):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought for user ID {agent_id}###\n\n{reasoning_text}\n\n")

def extract_json(text):
    """Extracts all JSON arrays from the LLM response, handling both single and multiple blocks."""
    
    # Find all JSON arrays enclosed within triple backticks
    matches = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', text, re.DOTALL)

    if not matches:
        # Fallback: find JSON arrays without triple backticks
        matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)

    if not matches:
        print("⚠️ No valid JSON arrays found in the response.")
        return 'retry'  # Indicate to retry the activity generation

    combined_activities = []

    for json_text in matches:
        # Remove inline comments (e.g., // comment)
        json_text_cleaned = re.sub(r'//.*', '', json_text)
        try:
            data = json.loads(json_text_cleaned)
            if isinstance(data, list):
                combined_activities.extend(data)
            else:
                combined_activities.append(data)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON block: {e}")
            continue

    return combined_activities if combined_activities else None

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, buffer_size=5):
    """Generates a bank log with multiple fraudulent and legitimate agents using a memory-efficient buffer."""

    fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
    legitimate_strategies = load_existing_strategies("strategies/legitimate_strategies.json")

    global_clock = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    output_file = "outputs/bank_log.csv"
    log_text_file="outputs/llm_chain_of_thought.txt"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Step 1: Initialize a fresh CSV and txt by clearing any existing content
    with open(output_file, 'w') as f:
        f.write("")  # Clear existing file content
    with open(log_text_file, 'w') as f:
        f.write("")  # Clear existing file content
    header_written = False  # Track if header is written
    buffer = []  # Initialize buffer

    # Helper function to write buffer to CSV and clear it
    def flush_buffer():
        nonlocal header_written
        if buffer:
            df = pd.DataFrame(buffer)
            # Ensure only expected columns are present
            expected_columns = [
                "transaction_id", "bank_timestamp", "local_timestamp", "user_id", "account_id",
                "type", "amount", "currency", "balance_before", "balance_after",
                "location", "ip_address", "device_id", "network_type", "merchant_name",
                "recipient_id", "recipient_bank", "granted", "login_attempts", "session_id",
                "velocity", "distance_from_last_location", "is_repeat_location",
                "fraud_label", "behavior_type"
            ]

            # Drop any unexpected columns
            df = df[[col for col in df.columns if col in expected_columns]]
            # Append to the CSV; write header only once
            df.to_csv(output_file, mode='a', index=False, header=not header_written)
            header_written = True  # Set header_written to True after the first write
            buffer.clear()  # Clear buffer after writing

    # Generate fraudulent activities
    fraud_activities_count = 0
    for _ in range(fraud_agents_count):
        user_id = str(uuid.uuid4())
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]

        activities = generate_activity_sequence(strategy=strategy, fraud_label=1, profile_type=behavior_type, global_clock=global_clock, user_id=user_id, log_text_file=log_text_file)
        for activity in activities:
            buffer.append(activity)
            fraud_activities_count += 1

            if len(buffer) >= buffer_size:
                flush_buffer()

            if fraud_activities_count >= int(total_activities * target_fraud_percentage):
                break
        if fraud_activities_count >= int(total_activities * target_fraud_percentage):
            break

    # Generate legitimate activities
    while fraud_activities_count + len(buffer) < total_activities:
        for _ in range(legit_agents_count):
            user_id = str(uuid.uuid4())
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]

            activities = generate_activity_sequence(strategy=strategy, fraud_label=0, profile_type=behavior_type, global_clock=global_clock, user_id=user_id)
            for activity in activities:
                buffer.append(activity)

                if len(buffer) >= buffer_size:
                    flush_buffer()

                if fraud_activities_count + len(buffer) >= total_activities:
                    break
            if fraud_activities_count + len(buffer) >= total_activities:
                break

    # Final flush to write any remaining activities in the buffer
    flush_buffer()

    print(f"✅ Activity generation complete for. Data saved to {output_file}")
    print(f"Required {total_activities} activities. Generated {fraud_activities_count + len(buffer)} activities.")
    print(f"Fraudulent activities: {fraud_activities_count}, Legitimate activities: {len(buffer) - fraud_activities_count}")
    print(f"Target fraud percentage: {target_fraud_percentage * 100}%")
    print(f"Datset fraud")

    # Load final dataframe (optional, if needed for further processing)
    final_df = pd.read_csv(output_file)
    return final_df


def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}


generate_activities(total_activities=10, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)