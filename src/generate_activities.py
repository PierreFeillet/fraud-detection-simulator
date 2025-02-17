from datetime import datetime, timedelta, timezone
import time
import json
import os
import random
import re
import uuid
from IPython import embed
import pandas as pd
import ollama
import matplotlib.pyplot as plt

# LLM used for sequnce generation
LLM_model = 'mistral'
# File paths
ERROR_LOG_FILE = f"outputs/json_errors_{LLM_model}.log"
REWARD_LOG_FILE = f"outputs/reward_progress_{LLM_model}.csv"
VALIDATION_LOG_FILE = f"outputs/json_validation_{LLM_model}.log"
CORRECTION_LOG_FILE = f"outputs/json_corrections_{LLM_model}.log"
log_text_file = f"outputs/llm_chain_of_thought_{LLM_model}.txt"
output_file = f"outputs/bank_log_{LLM_model}.csv"

# Expected field types for the JSON schema
EXPECTED_FIELD_TYPES = {
    "transaction_id": str,
    "bank_timestamp": str,  # ISO 8601 format
    "local_timestamp": str,  # ISO 8601 format
    "user_id": str,
    "account_id": str,
    "type": str,
    "amount": (int, float),  # Allow both int and float for amounts
    "currency": str,
    "balance_before": (int, float),
    "balance_after": (int, float),
    "location": str,
    "ip_address": str,
    "device_id": str,
    "network_type": str,
    "merchant_name": (str, type(None)),  # Can be string or null
    "recipient_id": (str, type(None)),  # Can be string or null
    "recipient_bank": (str, type(None)),  # Can be string or null
    "granted": bool,
    "login_attempts": int,
    "session_id": str,
    "velocity": (int, float),
    "distance_from_last_location": (int, float),
    "is_repeat_location": bool,
    "fraud_label": int,
    "behavior_type": str
}


# Initialize reward tracking
def initialize_logs():
    with open(REWARD_LOG_FILE, 'w') as file:
        file.write("timestamp,reward\n")
    with open(ERROR_LOG_FILE, 'w') as file:
        file.write("")
    with open(log_text_file, 'w') as file:
        file.write("")
    with open(output_file, 'w') as file:
        file.write("")

def update_reward_log(score, reason=""):
    """Logs reward score and reason for tracking performance of JSON generation."""
    log_entry = f"{datetime.now().isoformat()},{score},{reason}\n"
    with open(REWARD_LOG_FILE, 'a') as file:
        file.write(log_entry)
    
    print(f"🔹 Reward Logged: Score={score} | Reason: {reason}")



def visualize_rewards():
    if not os.path.exists(REWARD_LOG_FILE):
        print("No reward log found.")
        return
    df = pd.read_csv(REWARD_LOG_FILE)
    df['cumulative_reward'] = df['reward'].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['cumulative_reward'], marker='o', linestyle='-', label='Cumulative Reward')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward Progress Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def log_json_error(error_message, raw_output):
    with open(ERROR_LOG_FILE, "a") as log_file:
        log_file.write(f"Error: {error_message}\nOutput: {raw_output}\n{'-'*80}\n")
    

def extract_json(text, user_id):
    """Extracts JSON arrays from LLM responses and validates/corrects them before parsing."""
    matches = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', text, re.DOTALL) 

    if not matches:
        matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)  # Try finding inline JSON without triple backticks   
    
    if not matches:
        log_json_error("❌ No valid JSON arrays found in the response.", text)
        update_reward_log(score=-1, reason="No valid JSON found")
        return 'retry'

    # ✅ Step 1: Validate the raw extracted JSON
    if validate_json(matches):
        print(f"✅ JSON validation successful for user {user_id}") 
        json_corrected = False  # JSON is correct
    else:
        print(f"❌ Validation failed for user {user_id}. Attempting correction...") 
        # 🔄 Step 2: Attempt Correction
        corrected_json = correct_json(matches)
        if corrected_json and validate_json(corrected_json):
            print(f"✅ JSON successfully corrected for user {user_id}")
            matches = corrected_json
            json_corrected = True  # Flag to indicate JSON was fixed
        else:
            print(f"❌ Correction failed for user {user_id}. Returning 'retry' for regeneration...")
            update_reward_log(score=-1, reason="Correction failed")
            return 'retry'
    

    combined_activities = []
    for match in matches:
        try:
            data = json.loads(match)
            combined_activities.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error for user {user_id}: {e}")
            log_json_error(f"JSON Decode Error: {e}", match)
            update_reward_log(score=-1, reason="JSON decoding error")
            return 'retry'

    # ✅ Step 3: Check if the DataFrame can be filled
    df = pd.DataFrame(combined_activities)
    if not df.empty:
        reward_score = 2 if not json_corrected else 1  # +2 for perfect, +1 if corrected
        update_reward_log(score=reward_score, reason="JSON valid and filled DataFrame")
        return combined_activities
    else:
        print(f"⚠️ No activities found after extraction for user {user_id}. Retrying...")
        update_reward_log(score=-1, reason="Empty extracted JSON")
        return 'retry'


def log_json_validation(status, json_content, errors=None):
    """Logs JSON validation status and errors (if any) to a dedicated file."""
    log_file = "outputs/json_validation.log"
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"--- JSON Validation Result ---\n")
        file.write(f"Status: {status}\n")
        file.write(f"JSON:\n{json_content}\n")

        if errors:
            file.write("Errors:\n")
            for error in errors:
                file.write(f"- {error}\n")
        
        file.write("\n" + "="*80 + "\n")  # Separator for readability

def log_correction_result(corrected_json, original_json):
    """Logs JSON correction results for tracking improvements."""
    with open(CORRECTION_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
        log_file.write("Original JSON:\n")
        log_file.write(original_json + "\n")
        log_file.write("Corrected JSON:\n")
        log_file.write(corrected_json + "\n")
        log_file.write("-" * 80 + "\n\n")

def validate_json(activity_sequence, user_id):
    """Validates JSON correctness: required fields, correct types, and logical consistency."""
    errors = []

    if not activity_sequence:
        errors.append("❌ JSON is empty.")

    for activity in activity_sequence:
        transaction_id = activity.get("transaction_id", "UNKNOWN")
        
        # ✅ 1. Check for missing or extra fields
        missing_fields = [field for field in EXPECTED_FIELD_TYPES if field not in activity]
        extra_fields = [field for field in activity if field not in EXPECTED_FIELD_TYPES]
        
        if missing_fields:
            errors.append(f"❌ Missing fields in transaction {transaction_id}: {', '.join(missing_fields)}")
        if extra_fields:
            errors.append(f"❌ Unexpected fields in transaction {transaction_id}: {', '.join(extra_fields)}")
        
        # ✅ 2. Check field types
        for field, expected_type in EXPECTED_FIELD_TYPES.items():
            if field in activity and not isinstance(activity[field], expected_type):
                errors.append(
                    f"❌ Incorrect type for `{field}` in transaction {transaction_id}. "
                    f"Expected {expected_type}, got {type(activity[field])}."
                )

        # ✅ 3. Validate balance computations
        transaction_type = activity.get("type", "").lower()
        balance_before = activity.get("balance_before")
        balance_after = activity.get("balance_after")
        amount = activity.get("amount", 0)
        granted = activity.get("granted")

        if balance_before is None or balance_after is None:
            errors.append(f"❌ Missing balance fields in transaction {transaction_id}.")
            continue

        if transaction_type in ["deposit", "transfer_in"]:
            expected_balance_after = balance_before + amount
            if balance_after != expected_balance_after or granted is not True:
                errors.append(
                    f"❌ Incorrect balance for {transaction_type} in transaction {transaction_id}. "
                    f"Expected balance_after={expected_balance_after}, got {balance_after}."
                )

        elif any(word in transaction_type for word in ["purchase", "withdrawal", "transfer"]):
            if amount <= balance_before:  # ✅ Transaction should be granted
                expected_balance_after = balance_before - amount
                if balance_after != expected_balance_after or granted is not True:
                    errors.append(
                        f"❌ Incorrect balance for {transaction_type} in transaction {transaction_id}. "
                        f"Expected balance_after={expected_balance_after}, got {balance_after}."
                    )
            else:  # ❌ Insufficient funds → should be denied
                if balance_after != balance_before or granted is not False:
                    errors.append(
                        f"❌ Insufficient funds for {transaction_type} in transaction {transaction_id}. "
                        f"Expected granted=False, balance_after={balance_before}, got granted={granted}, balance_after={balance_after}."
                    )

        elif transaction_type in ["login", "authentication"]:
            if balance_after != balance_before:
                errors.append(
                    f"❌ Non-transaction activity '{transaction_type}' should not modify balance in transaction {transaction_id}."
                )
    # 4. Log and return validation results
    validation_status = "VALID" if not errors else "INVALID"
    
    if validation_status == "VALID":
        update_reward_log(success=True)
        print(f"✅ JSON Validation: {validation_status}")
        log_json_validation(validation_status, json.dumps(activity_sequence, indent=2))
        return True
    else:
        update_reward_log(success=False)
        print(f"❌ JSON Validation: {validation_status}")
        log_json_validation(validation_status, json.dumps(activity_sequence, indent=2), errors)
        return False


def correct_json(activity_sequence):
    """Uses a small LLM to correct minor JSON errors."""
    json_string = json.dumps(activity_sequence, indent=2)

    correction_prompt = f"""
    You are an expert JSON corrector. Here is a JSON that contains errors:
    
    ```json
    {json_string}
    ```
    
    Your task:
    - Fix any structural issues.
    - Ensure numerical consistency.
    - Return ONLY the corrected JSON, nothing else.
    """

    response = ollama.chat(model="phi", messages=[{"role": "user", "content": correction_prompt}])
    corrected_json = response['message']['content'].strip()

    # ✅ Log correction results
    log_correction_result(corrected_json, json_string)

    return corrected_json


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
        f"- The JSON example is just a support for you to understand the structure I want. You must fill the fields with the most appropriate values for the strategy you are simulating.\n"
    )


def generate_activity_sequence(strategy, fraud_label=0, profile_type="Legitimate", global_clock=None, user_id=None, max_retries=3):
    """Generates structured financial activities based on the provided strategy with retry limit."""

    retries = 0  # Track retry attempts

    while retries < max_retries:
        prompt = build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id)
        response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}])
        raw_response = response['message']['content'].strip()

        save_to_text(raw_response, user_id)

        activity_sequence = extract_json(raw_response, user_id)

        if activity_sequence != 'retry':
            print(f"✅ Activity sequence generated successfully for user ID {user_id} on attempt {retries + 1}")
            return activity_sequence  

        print(f"🔄 Retrying activity sequence due to invalid JSON... (Attempt {retries + 1} of {max_retries})")
        retries += 1

    print(f"❌ Failed to generate valid activity sequence for user {user_id} after {max_retries} retries.")
    return 'retry'  # Signal failure


def save_to_text(reasoning_text, agent_id=None):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(log_text_file, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought for user ID {agent_id} ###\n\n{reasoning_text}\n\n")

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, buffer_size=3):
    """Generates a bank log with multiple fraudulent and legitimate agents using a memory-efficient buffer."""
    fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
    legitimate_strategies = load_existing_strategies("strategies/legitimate_strategies.json")

    global_clock = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    header_written = False  
    buffer = []  

    def flush_buffer():
        nonlocal header_written
        if buffer:
            df = pd.DataFrame(buffer)
            expected_columns = list(EXPECTED_FIELD_TYPES.keys())
            df = df[[col for col in df.columns if col in expected_columns]]
            df.to_csv(output_file, mode='a', index=False, header=not header_written)
            header_written = True  
            buffer.clear()  

    fraud_activities_count = 0
    for _ in range(fraud_agents_count):
        user_id = str(uuid.uuid4())
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]

        activities = generate_activity_sequence(strategy=strategy, fraud_label=1, profile_type=behavior_type, global_clock=global_clock, user_id=user_id)
        for activity in activities:
            buffer.append(activity)
            fraud_activities_count += 1

            if len(buffer) >= buffer_size:
                flush_buffer()

            if fraud_activities_count >= int(total_activities * target_fraud_percentage):
                break
        if fraud_activities_count >= int(total_activities * target_fraud_percentage):
            break

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

    flush_buffer()

    print(f"✅ Activity generation complete. Data saved to {output_file}")
    print(f"Total activities: {fraud_activities_count + len(buffer)}")
    print(f"Fraudulent activities: {fraud_activities_count}")
    print(f"Legitimate activities: {len(buffer) - fraud_activities_count}")
    print(f"Fraud percentage: {target_fraud_percentage * 100}%")

    final_df = pd.read_csv(output_file)
    return final_df

def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

start_time = time.time()
print(f"Simulation started at {datetime.now().isoformat()}")
initialize_logs()
generate_activities(total_activities=20, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)
# Time required to generate activities in minutes approximated
time_taken = round(time.time()-start_time/60,2)
print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")  
visualize_rewards()