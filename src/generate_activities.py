from collections import defaultdict
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
OUTPUT_DIR = "outputs"
ERROR_TRACKING_FILE = os.path.join(OUTPUT_DIR, f"error_tracking_{LLM_model}.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_errors_{LLM_model}.log")
REWARD_LOG_FILE = os.path.join(OUTPUT_DIR, f"reward_progress_{LLM_model}.csv")
VALIDATION_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_validation_{LLM_model}.log")
LOG_TEXT_FILE = os.path.join(OUTPUT_DIR, f"llm_chain_of_thought_{LLM_model}.txt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"bank_log_{LLM_model}.csv")

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_logs():
    """Initializes all log files."""
    for file_path in [REWARD_LOG_FILE, ERROR_LOG_FILE, LOG_TEXT_FILE, OUTPUT_FILE, VALIDATION_LOG_FILE, ERROR_TRACKING_FILE]:
        with open(file_path, 'w') as file:
            file.write("")

def log_error_occurrences(errors):
    """Logs and tracks how often each type of error occurs over time."""
    
    # Load existing error records if the file exists
    if os.path.exists(ERROR_TRACKING_FILE):
        with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as file:
            error_data = json.load(file)
    else:
        error_data = {"error_counts": defaultdict(int), "timestamps": []}

    # Update error counts
    for error in errors:
        error_data["error_counts"][error] = error_data["error_counts"].get(error, 0) + 1

    # Track timestamp of latest error set
    error_data["timestamps"].append(datetime.now().isoformat())

    # Save back to the file
    if os.path.exists(ERROR_TRACKING_FILE) and os.path.getsize(ERROR_TRACKING_FILE) > 0:
        with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as file:
            try:
                error_data = json.load(file)
            except json.JSONDecodeError:
                print("⚠️ Warning: Corrupt or empty error tracking file. Resetting.")
                error_data = {"error_counts": defaultdict(int), "timestamps": []}
    else:
        error_data = {"error_counts": defaultdict(int), "timestamps": []}

def validate_json(raw_response, user_id):
    """Validates JSON correctness and returns a list of detected errors while logging occurrences.
    Returns 'retry' if the JSON is invalid and needs to be reattempted."""  
    errors = []
    print(f"Validating JSON for user {user_id}...")
    extracted_json_blocks = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', raw_response, re.DOTALL) 

    if not extracted_json_blocks:
        extracted_json_blocks = re.findall(r'(\[\s*{.*?}\s*\])', raw_response, re.DOTALL)

    if not extracted_json_blocks:
        errors.append(f"Error: JSON not enclosed in triple backticks or missing valid structure.")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}:\n{errors}")
        return 'retry'
    else:
        valid_json_strings = []

        for json_string in extracted_json_blocks:
            # 🔹 Step 1: Check for forbidden inline comments before parsing
            if re.search(r'//.*', json_string) or re.search(r'/\*.*?\*/', json_string, re.DOTALL):
                errors.append("Error: JSON contains inline comments (`//` or `/*...*/`). Remove eventual comments.")
                log_json_errors(errors)
                print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                return 'retry'
            # 🔹 Step 2: Store for later parsing if no comments are found
            valid_json_strings.append(json_string)

        valid_activities = []
        for json_string in valid_json_strings:
            try:
                activities = json.loads(json_string)  # Convert JSON string to Python object
                if not isinstance(activities, list):
                    activities = [activities]

                # 🔹 Step 3: Validate JSON structure and field types
                for activity in activities:
                    missing_fields = [field for field in EXPECTED_FIELD_TYPES if field not in activity]
                    extra_fields = [field for field in activity if field not in EXPECTED_FIELD_TYPES]

                    if missing_fields:
                        errors.append(f"Error:  Missing fields in transaction: {', '.join(missing_fields)}")
                    if extra_fields:
                        errors.append(f"Error: Unexpected fields in transaction: {', '.join(extra_fields)}")

                    for field, expected_type in EXPECTED_FIELD_TYPES.items():
                        if field in activity and not isinstance(activity[field], expected_type):
                            errors.append(
                                f"Error: Incorrect type for `{field}` in transaction. "
                                f"Error: Expected {expected_type}, got {type(activity[field])}."
                            )
                    
                    if check_balance(activity, errors):
                        valid_activities.append(activity)

            except json.JSONDecodeError as e:
                errors.append(f"JSON Decode Error: {e}")

    if not errors:
        print(f"JSON validated successfully for user {user_id}.")
        #update_reward_log()
        return valid_activities
    else:
        #log_error_occurrences(errors)
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}:\n{errors}")
        #update_reward_log()
        return 'retry'


def check_balance(activity, errors):
                transaction_type = activity.get("type", "").lower()
                balance_before = activity.get("balance_before")
                balance_after = activity.get("balance_after")
                amount = activity.get("amount", 0)
                granted = activity.get("granted")
                if balance_before is None or balance_after is None or amount is None:
                    errors.append(f"Error:  Missing balance fields and amount in transaction.")
                    return False

                if any(word in transaction_type for word in ["purchase", "withdrawal", "transfer"]):
                    if amount > balance_before:
                        if granted:
                            errors.append(
                                f"Error:  Insufficient funds for {transaction_type} in transaction. "
                                f"Error: Expected granted=False, balance_after={balance_before}, got granted={granted}, balance_after={balance_after}."
                            )
                            return False
                    else:
                        expected_balance_after = balance_before - amount
                        if balance_after != expected_balance_after:
                            errors.append(
                                f"Error: Incorrect balance for {transaction_type} in transaction. "
                                f"Error: Expected balance_after={expected_balance_after}, got {balance_after}."
                            )
                        return False
                return True

def plot_error_trends():
    """Visualizes the frequency of different errors over time."""
    if not os.path.exists(ERROR_TRACKING_FILE):
        print("No error tracking data found.")
        return

    with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as file:
        error_data = json.load(file)

    if not error_data["error_counts"]:
        print("No errors recorded yet.")
        return

    errors = list(error_data["error_counts"].keys())
    counts = list(error_data["error_counts"].values())

    plt.figure(figsize=(12, 6))
    plt.barh(errors, counts)
    plt.xlabel("Occurrences")
    plt.ylabel("Error Type")
    plt.title("Error Occurrences Over Time")
    plt.tight_layout()
    plt.show()

def update_reward_log(score=1):
    """Logs reward score based on JSON correctness and recurring error patterns."""
    
    # Load error tracking data
    if os.path.exists(ERROR_TRACKING_FILE):
        with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as file:
            error_data = json.load(file)
            total_errors = sum(error_data["error_counts"].values())  # Count total errors
    else:
        total_errors = 0

    # Adjust reward: More errors reduce score
    adjusted_score = score - (total_errors * 0.1)  # Penalize based on error occurrences

    log_entry = f"{datetime.now().isoformat()},{adjusted_score}\n"
    if os.path.exists(ERROR_TRACKING_FILE) and os.path.getsize(ERROR_TRACKING_FILE) > 0:
        with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as file:
            try:
                error_data = json.load(file)
            except json.JSONDecodeError:
                print("⚠️ Warning: Resetting corrupted error tracking file.")
                error_data = {"error_counts": {}}
    else:
        error_data = {"error_counts": {}}


    print(f"🔹 Reward Logged: Adjusted Score={adjusted_score}")

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

def log_json_errors(error_array):
    """Logs an array of errors to a JSON file."""
    
    # Ensure the file exists and read existing data
    existing_errors = []
    if os.path.exists(ERROR_LOG_FILE):
        try:
            with open(ERROR_LOG_FILE, "r") as log_file:
                existing_errors = json.load(log_file)
        except json.JSONDecodeError:
            existing_errors = []  # Reset if corrupted
    
    # Append new errors to existing ones
    existing_errors.extend(error_array)

    # Write updated error log
    with open(ERROR_LOG_FILE, "a") as log_file:
        json.dump(existing_errors, log_file, indent=2)

def read_past_errors():
    """Reads past errors from the log file and returns them as an array."""
    if not os.path.exists(ERROR_LOG_FILE):
        return []  # Return empty if no log file exists
    
    try:
        with open(ERROR_LOG_FILE, "r") as log_file:
            past_errors = json.load(log_file)  # Load JSON array
        return past_errors
    except json.JSONDecodeError:
        return []  # Return empty array if JSON is malformed

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

def build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id, past_errors=None):
    """Builds the prompt for generating activity sequences, adapting based on past errors."""
    
    error_feedback = "### Errors you MUST avoid:\n" + "\n".join([f"- {err}" for err in past_errors]) if past_errors else ""
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
    # Convert expected field types into a readable format for the LLM
    field_types_description = "\n".join(
        [f"- `{field}`: {str(expected_type)}" for field, expected_type in EXPECTED_FIELD_TYPES.items()]
    )
    # Final Prompt with Strategy and JSON Example
    return (
    f"You are an AI generating **detailed sequences of banking activities** for a fraud simulation.\n"
    f"Your task is to create a sequence of financial activities based on the predefined strategy below.\n\n"

    f"### Strategy:\n{strategy}\n\n"

    f"If `{fraud_label}`=1: "
    f"  1. Ensure the sequence of activities **completes the fraud** as described in the strategy by the minimum number of activities."
    f"  2. If the fraud is a **multi-event fraud**, generate at least **`minimum_activities`** (specified in the {strategy}) transactions that follow the specified pattern."
    f"  3. For **single-event fraud**, ensure the transaction fully represents the fraudulent behavior."
    f"  4. The generated activities must clearly reflect the fraud type, scope, and behavior characteristics provided in the strategy.\n\n"
    f"{error_feedback}\n\n"

    f"### Data Generation Rules:\n"
    f"The financial activities must be generated as a JSON object.\n\n"

    f"### Required Fields in the JSON (with format specifications, do NOT add extra fields):\n{field_explanation}\n\n"

    f"- First, explain your reasoning step by step **without making examples**.\n"
    f"- Then, generate **only** the structured JSON activity sequence as in the following JSON example: \n{json_template}\n"
    f"- The JSON MUST be enclosed within **triple backticks** using the format ```json ... ```.\n"
    f"- Ensure that timestamps are logically consistent and formatted according to ISO 8601 standards.\n"
    f"- Do **NOT** include comments or explanations in the JSON.\n"
    f"- End your response immediately after closing the JSON block.\n"
    f"- **Use the provided `fraud_label`, `profile_type`, `global_clock`, `user_id`. They must remain exactly as given: `{fraud_label},{profile_type}, {global_clock},{user_id}`.**\n\n"

    f"### Important:\n"
    f"- Ensure that the JSON is correctly formatted, with the correct formats.\n"
    f"- **The JSON must be flat**\n"
    f"- End your response after closing triple backticks.\n"
    f"- Ensure all field values match the types and formats specified in the JSON field explanation. If the format is string, the string must be enclosed in \"\" or ''\n"
    f"- Don't repeat the JSON example, just fill the fields with the most appropriate values for the strategy you are simulating.\n\n"
    f"** Field Type Constraints (MUST be followed):**\n{field_types_description}"
    )

#Add usefyul prints during build_generation_prompt
#- need to understand if the errors are beinng used and in case of invali structure it must be return retry


def generate_activity_sequence(strategy, fraud_label=0, profile_type="Legitimate", global_clock=None, user_id=None, max_retries=3):   
    """Generates structured financial activities with adaptive learning."""
    retries = 1
    past_errors = []
    while retries <= max_retries:
        print(f"Attempt {retries}/{max_retries} to generate activities for user {user_id}...")
        prompt = build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id, past_errors)
        response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}], temperature=0.6)    
        raw_response = response['message']['content'].strip()
        save_to_text(raw_response, user_id)  # Save for debugging
        activity_sequence = validate_json(raw_response, user_id)
        # 🔹 Step 2: Extract, Validate, and Correct JSON
        if activity_sequence != 'retry':
            return activity_sequence 
        else:
            past_errors = read_past_errors()
            print(f"Retrying activity generation for user {user_id}...")
            #embed()
            retries += 1
    

def visualize_json_success_rate():
    """Plots JSON success trends over time."""
    if not os.path.exists(REWARD_LOG_FILE):
        return
    df = pd.read_csv(REWARD_LOG_FILE)
    df['cumulative_success'] = (df['reward'] == 2).cumsum()
    plt.plot(df.index, df['cumulative_success'], marker='o')
    plt.xlabel('Attempts')
    plt.ylabel('Valid JSONs')
    plt.title('JSON Success Rate')
    plt.grid()
    plt.show()


def save_to_text(reasoning_text, agent_id=None):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(LOG_TEXT_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n### LLM Chain of Thought for user ID {agent_id} ###\n\n{reasoning_text}\n\n")

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, buffer_size=3):
    """Generates a bank log with multiple fraudulent and legitimate agents using a memory-efficient buffer."""
    fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
    legitimate_strategies = load_existing_strategies("strategies/legitimate_strategies.json")

    global_clock = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

    header_written = False  
    buffer = []  
    
    def flush_buffer():
        nonlocal header_written
        if buffer:
            df = pd.DataFrame(buffer)
            expected_columns = list(EXPECTED_FIELD_TYPES.keys())
            df = df[[col for col in df.columns if col in expected_columns]]
            df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not header_written)
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

    print(f"Activity generation complete. Data saved to {OUTPUT_FILE}")
    print(f"Total activities: {fraud_activities_count + len(buffer)}")
    print(f"Fraudulent activities: {fraud_activities_count}")
    print(f"Legitimate activities: {len(buffer) - fraud_activities_count}")
    print(f"Fraud percentage: {target_fraud_percentage * 100}%")

    final_df = pd.read_csv(OUTPUT_FILE)
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
visualize_json_success_rate()
plot_error_trends()

visualize_rewards()