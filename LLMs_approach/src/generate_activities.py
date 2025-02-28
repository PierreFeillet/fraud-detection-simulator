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
import secrets

from watsonx_helper import watsonx_chat

# LLM used for sequnce generation
LLM_model = 'deepseek-r1'
# File paths
OUTPUT_DIR = "outputs"
ERROR_TRACKING_FILE = os.path.join(OUTPUT_DIR, f"error_tracking_{LLM_model}.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_errors_{LLM_model}.log")
REWARD_LOG_FILE = os.path.join(OUTPUT_DIR, f"reward_progress_{LLM_model}.csv")
VALIDATION_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_validation_{LLM_model}.log")
LOG_TEXT_FILE = os.path.join(OUTPUT_DIR, f"llm_chain_of_thought_{LLM_model}.txt")
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, f"bank_log_{LLM_model}.csv")

# Expected field types for the JSON schema
# Expected field types for the JSON schema
EXPECTED_FIELD_TYPES = {
    "bank_timestamp": str,  # ISO 8601 format
    "local_timestamp": str,  # ISO 8601 format
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
    "login_attempts": int,
    "session_id": str,
    "velocity": (int, float),
    "distance_from_last_location": (int, float),
    "is_repeat_location": bool,
}


ORDERED_COLUMNS = [
    "transaction_id",  # Transaction ID first
    "user_id",
    
    # Time-related fields together
    "bank_timestamp",
    "local_timestamp",

    # Geographical information together
    "location",
    "ip_address",
    "device_id",
    "network_type",

    # Transaction-related fields grouped
    "type",
    "amount",
    "currency",
    "account_id",
    "balance_before",
    "balance_after",

    # Recipient details
    "merchant_name",
    "recipient_id",
    "recipient_bank",

    # Behavior and fraud information
    "behavior_type",
    "fraud_label"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ERROR_TRACKING_FILE
def initialize_logs():
    """Initializes all log files."""
    for file in [REWARD_LOG_FILE, ERROR_LOG_FILE, LOG_TEXT_FILE, DATA_FILE, VALIDATION_LOG_FILE, ]:
        with open(file, 'w') as file:
            file.write("")
    # To ensure the csv is restarted
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)


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
    extracted_json_blocks = re.findall(r'```\s*(\[\s*{.*?}\s*\])\s*```end_json', raw_response, re.DOTALL) 

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
                        log_json_errors(errors)
                        print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                        return 'retry'

                    if extra_fields:
                        errors.append(f"Error: Unexpected fields in transaction: {', '.join(extra_fields)}")
                        log_json_errors(errors)
                        print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                        return 'retry'


                    for field, expected_type in EXPECTED_FIELD_TYPES.items():
                        if field in activity and not isinstance(activity[field], expected_type):
                            errors.append(
                                f"Error: Incorrect type for `{field}` in transaction. "
                                f"Error: Expected {expected_type}, got {type(activity[field])}."
                            )
                            log_json_errors(errors)
                            print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                            return 'retry'
  
                    if check_balance(activity, errors):
                        valid_activities.append(activity)

            except json.JSONDecodeError as e:
                errors.append(f"JSON Decode Error: {e}")
                log_json_errors(errors)
                print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                return 'retry'


    if not errors:
        print(f"JSON validated successfully for user {user_id}.")
        return valid_activities

def check_balance(activity, errors):
                transaction_type = activity.get("type", "").lower()
                balance_before = activity.get("balance_before")
                balance_after = activity.get("balance_after")
                amount = activity.get("amount")
                if balance_before is None or balance_after is None or amount is None:
                    error= f"Error: `balance_before`, `amount`, `balance_after` fields are None"
                    errors.append(error)
                    log_json_errors(errors)
                    print(errors)
                    return 'retry'

                if any(word in transaction_type for word in ["purchase", "withdrawal", "transfer"]):
                    if amount > balance_before:
                        if balance_after!=balance_before:
                            error = f"Error: Insufficient funds for {transaction_type} in transaction. `balance_after` must be = `balance_before`."
                            errors.append(error)
                            log_json_errors(errors)
                            print(errors)
                            return 'retry'

                    else:
                        expected_balance_after = balance_before - amount
                        if balance_after != expected_balance_after:
                            error = f"Error: Incorrect balance for {transaction_type} in transaction. Error: Expected balance_after={expected_balance_after}, got {balance_after} for transaction type {transaction_type}."
                            errors.append(error)
                            log_json_errors(errors)
                            print(errors)
                            return 'retry'
                else:
                    expected_balance_after = balance_before+amount
                    if balance_after != expected_balance_after:
                        error = f"Error: Incorrect balance for {transaction_type} in transaction. Error: Expected balance_after={expected_balance_after}, got {balance_after} for transaction type {transaction_type}."
                        errors.append(error)
                        log_json_errors(errors)
                        print(errors)   
                        return
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

# Initialize reward tracking
def initialize_reward_log():
    if not os.path.exists(REWARD_LOG_FILE):
        with open(REWARD_LOG_FILE, 'w') as file:
            file.write("timestamp,user_id,reward,reason\n")

def update_reward_log(score: int, user_id: str, reason: str):
    with open(REWARD_LOG_FILE, 'a') as file:
        file.write(f"{datetime.now().isoformat()},{user_id},{score},{reason}\n")



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
    """Logs an array of errors to a JSON file, storing each array as a separate JSON line."""
    with open(ERROR_LOG_FILE, "a") as log_file:
        json.dump(error_array, log_file)
        log_file.write("\n")  # Newline to separate each JSON array


def read_past_errors():
    """Reads past errors from the log file and returns them as a combined list."""
    if os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, 'r') as file:
            try:
                lines = file.readlines()
                all_errors = []
                for line in lines:
                    try:
                        errors = json.loads(line.strip())  # Load each line as a JSON array
                        all_errors.extend(errors)
                    except json.JSONDecodeError:
                        continue
                return list(set(all_errors))  # Remove duplicates
            except Exception as e:
                print(f"⚠️ Error reading past errors: {e}")
                return []
    return []


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

def build_generation_prompt(strategy, global_clock, user_id, past_errors=None):
    """Builds the prompt for generating activity sequences, adapting based on fraud type and past errors."""

    # JSON Template with Dynamic Values
    json_template = """
    ```
        [
        {
            "bank_timestamp": "2025-03-01T07:00:00+00:00",
            "local_timestamp": "2025-03-01T02:00:00-05:00",
            "account_id": "ACC-12345678",
            "type": "Purchase",
            "amount": 150.75,
            "currency": "USD",
            "balance_before": 1000.00,
            "balance_after": 849.25,
            "location": "New York, USA",
            "ip_address": "192.168.1.10",
            "device_id": "iPhone-14",
            "network_type": "Wi-Fi",
            "merchant_name": "Amazon",
            "recipient_id": null,
            "recipient_bank": null,
            "login_attempts": 1,
            "session_id":"S-12345678",
            "velocity": 0.52,
            "distance_from_last_location": 3.4,
            "is_repeat_location": true,
        },
        {
            "bank_timestamp": "2025-03-01T07:10:00+00:00",
            "local_timestamp": "2025-03-01T02:10:00-05:00",
            "account_id": "ACC-12345678",
            "type": "Transfer",
            "amount": 500.00,
            "currency": "USD",
            "balance_before": 849.25,
            "balance_after": 349.25,
            "location": "San Francisco, USA",
            "ip_address": "192.168.1.20",
            "device_id": "MacBook-Air",
            "network_type": "Wi-Fi",
            "merchant_name": null,
            "recipient_id": "REC-12345678",
            "recipient_bank": "NWBKGB2L",
            "login_attempts": 1,
            "session_id": "S-15467349",
            "velocity": 10.0,
            "distance_from_last_location": 4150.0,
            "is_repeat_location": false,
        }
        ]
    ```end_json
    """
    field_explanation = f""""
        - `bank_timestamp`: string (ISO 8601 format), The UTC time when the activity occurred starting from {global_clock}, formatted as 'YYYY-MM-DDTHH:MM:SS+00:00'.\n
        - `local_timestamp`: string (ISO 8601 format), The local time of the activity with time zone offset (e.g., 'YYYY-MM-DDTHH:MM:SS-05:00').\n
        - `account_id`: string, A structured account ID starting with **"ACC-"** followed by 8 digits (e.g., `"ACC-12345678"`). This is the target account for that activity.\n
        - `type`: string, The type of activity (e.g., Purchase, Withdrawal, Transfer).\n
        - `amount`: flaot, The monetary amount involved in the activity. If the activity deosn't involve money movement then `amount`=0 (example: `Login` ==> `amount`=0 )\n
        - `currency`: string, The currency of the transaction (e.g., 'USD', 'EUR').\n
        - `balance_before`: flaot, The account balance before the activity.\n
        - `balance_after`: flaot, The account balance after the activity transaction (adjusted based on the `amount` and the `type`). `balance_after`=`` If the computed `balance_after`<0 then `balance_after`=`balance_before`\n
        - `location`: string, The city and country where the transaction took place.\n
        - `ip_address`: string, The IP address used during the activity (IPv4 format, e.g., `"192.168.1.10"`).\n
        - `device_id`: string, The device identifier, typically the device model (e.g., `"iPhone-14"` or `"Pixel-7"`).\n
        - `network_type`: string, The network used during the transaction (e.g., `"Wi-Fi"`, `"Mobile Data"`).\n
        - `merchant_name`: string or null, The name of the merchant (e.g., `"Amazon"` or `"Walmart"`), otherwise null.\n
        - `recipient_id`: string or null, For transfers: structured like a bank account ID, starting with **"REC-"** followed by 8 digits (e.g., `"REC-98765432"`), or `null` if not applicable.\n
        - `recipient_bank`: string or null, For transfers: a simulated BIC code (8 or 11 alphanumeric characters, e.g., `"DEUTDEFFXXX"`), or `null` if not applicable.\n
        - `login_attempts`: int, The number of login attempts during the session.\n
        - `session_id`: string, A unique identifier for the session using "S-" followed by 8 digits (e.g., `"S-12345678"`).\n
        - `velocity`: float, Time difference in minutes between the current and previous activity.\n
        - `distance_from_last_location`: float, Distance in kilometers from the previous transaction location.\n
        - `is_repeat_location`: bool, True if the location is previously used, False otherwise.\n
    """
    # Convert expected field types into a readable format for the LLM
    field_types_description = "\n".join(
        [f"- `{field}`: {str(expected_type)}" for field, expected_type in EXPECTED_FIELD_TYPES.items()]
    )

    balance_rules = """
    - Rule for `balance_before`: check if the `account_id` is already present in the sequence. If not, initialize the `balance_before` to a reasonable number; if yes, `balance_before`=`balance_after` of the latest transaction for that account. \n
    - Rule for `balance_after`: if the activity `type` involves adding money on the account then compute `balance_after` as `balance_before - amount` and assign only the result value. if the activity `type` involves taking money from the account then compute `balance_after` as `balance_before - amount` and assign only the result value. If the activity `type` doesn't involve any impact on the balance for that account, then `balance_after`=`balance_before` for that account_id. \n"
    """
     # Final Prompt with Strategy and JSON Example
    return (
        f"You are an AI generating **detailed sequences of banking activities** for a fraud simulation.\n"  
        f"Your task is to create a sequence of financial activities based on the predefined strategy below.\n\n"
        f"### Strategy:\n{strategy}\n\n"
        f"- If the fraud is a **multi-event fraud**, generate at least **`minimum_activities`** (specified in the strategy) transactions that follow the specified pattern."
        f"- For **single-event fraud**, ensure the transaction fully represents the fraudulent behavior."
        f"The generated activities must clearly reflect the characteristics provided in the strategy.\n\n"
        f"### Data Generation Rules:\n"
        f"- Required Fields in the JSON, with format specifications, don't add extra-fields:\n{field_explanation}\n\n"
        f"- First, explain your reasoning step by step **without making examples**.\n"
        f"- Then, generate **only** the structured JSON activity sequence as in the following JSON example: \n{json_template}\n"
        f"- Don't repeat the example, generate a new one.\n"
        f"- The JSON MUST be enclosed within **triple backticks** and must have '```end_json' as a stop sequence.\n"
        f"- Ensure that timestamps are logically consistent and formatted according to ISO 8601 standards.\n"
        f"- Do **NOT** include comments or explanations in the JSON.\n"
        f"- End your response immediately after closing the JSON block.\n"
        f"- **Use the provided `global_clock` to iniziliaze the `bank_timestamp` for only the first activity in the sequence.**\n"
        f"- **Field Type Constraints (MUST be followed):** {field_types_description}"
        f"- To choose the value for `balance_before` and compute the `balance_after` follow the rules in Balance Rules section.\n"
        f"- Balance Rules: {balance_rules}\n"
        f"### Important:\n"
        f"- **The JSON must be flat**\n"
        f"- End your response after closing triple backticks and add ``.\n"
        f"- Ensure all field values match the types and formats specified in the JSON field explanation. If the format is string, the string must be enclosed in "" or ''\n"
        f"- Ensure the balance after is correctly calculated based on the transaction type and amount.\n"
        f"- Don't repeat the same errors you did in the past: {past_errors}.\n"
    )

#Add usefyul prints during build_generation_prompt
#- need to understand if the errors are beinng used and in case of invali structure it must be return retry


def generate_activity_sequence(strategy, global_clock=None, user_id=None,):   
    """Generates structured financial activities with adaptive learning."""
    retries = 1
    while True:
        past_errors = read_past_errors()
        print(f"Past errors: {past_errors}")
        print(f"Attempt {retries} to generate activities for user {user_id}...")
        prompt = build_generation_prompt(strategy, global_clock, user_id, past_errors)
        response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}], )    # options={"temperature": 0.7}
        raw_response = response['message']['content'].strip()
        #raw_response = watsonx_chat(prompt).strip()
        save_to_text(raw_response, user_id)  # Save for debugging
        activity_sequence = validate_json(raw_response, user_id)
        # 🔹 Step 2: Extract, Validate, and Correct JSON
        if activity_sequence != 'retry':
            print(f"JSON generation successful for user {user_id}.")
            update_reward_log(score=1, user_id=user_id, reason="Valid JSON extracted and used")
            return activity_sequence 
        else:
            print(f"Retrying activity generation for user {user_id}...")
            update_reward_log(score=-1, user_id=user_id, reason="JSON generation failed")
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

def generate_random_hash(length=8):
    """Generate a random hash of the specified length."""
    return secrets.token_hex(length // 2)  # Each hex character is 4 bits

def assign_actvity_fields(activity, user_id, behavior_type, fraud_label):   
    activity['user_id'] = user_id
    activity['transaction_id'] = f"TXN-{generate_random_hash(10)}"
    activity['behavior_type'] = behavior_type
    activity['fraud_label'] = fraud_label
    return activity

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, buffer_size=1):
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
            df = df[ORDERED_COLUMNS]
            #expected_columns = list(EXPECTED_FIELD_TYPES.keys())
            #df = df[[col for col in df.columns if col in expected_columns]]
            df.to_csv(DATA_FILE, mode='a', index=False, header=not header_written)
            header_written = True  
            buffer.clear()  

    fraud_activities_count = 0
    for _ in range(fraud_agents_count):
        user_id = f"USER-{generate_random_hash(8)}"
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]
        activities = generate_activity_sequence(strategy=strategy, global_clock=global_clock, user_id=user_id)
        for activity in activities:  
            assign_actvity_fields(activity, user_id, behavior_type, fraud_label=1)  
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
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]
            activities = generate_activity_sequence(strategy=strategy, global_clock=global_clock, user_id=user_id)
            for activity in activities:
                assign_actvity_fields(activity, user_id, behavior_type, fraud_label=0)  
                buffer.append(activity)

                if len(buffer) >= buffer_size:
                    flush_buffer()

                if fraud_activities_count + len(buffer) >= total_activities:
                    break
            if fraud_activities_count + len(buffer) >= total_activities:
                break

    flush_buffer()

    print(f"Activity generation complete. Data saved to {DATA_FILE}")

    final_df = pd.read_csv(DATA_FILE)
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
initialize_reward_log()
generate_activities(total_activities=20, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)
# Time required to generate activities in minutes approximated
time_taken = round(time.time()-start_time/60,2)
print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")  
visualize_json_success_rate()
plot_error_trends()

visualize_rewards()