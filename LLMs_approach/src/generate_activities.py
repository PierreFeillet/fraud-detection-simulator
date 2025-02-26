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
EXPECTED_FIELD_TYPES = {
    "bank_timestamp": str,  # ISO 8601 format
    "local_timestamp": str,  # ISO 8601 format
    "account_id": str,
    "type": str,
    "amount": (int, float),  # Allow both int and float for amounts
    "currency": str,
    "balance_before": (int, float),
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
    "account_id",
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
    "balance_before",
    "balance_after",
    "granted",

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
    for file_path in [REWARD_LOG_FILE, ERROR_LOG_FILE, LOG_TEXT_FILE, DATA_FILE, VALIDATION_LOG_FILE, ]:
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
                        
                    if activity.get("balance_before") is None:
                        error= f"Error: Missing value for balance_before."
                        errors.append(error)
                        log_json_errors(errors)
                        print(errors)
                        return 'retry'
                    
                    if activity.get("amount") is None:
                        error= f"Error: Missing value for amount."
                        errors.append(error)
                        log_json_errors(errors)
                        print(errors)
                        return 'retry'

                    valid_activities.append(activity)

            except json.JSONDecodeError as e:
                errors.append(f"JSON Decode Error: {e}")
                log_json_errors(errors)
                print(f"Errors detected in JSON for user {user_id}:\n{errors}")
                return 'retry'
    if not errors:
        print(f"JSON validated successfully for user {user_id}.")
        return valid_activities

def compute_balance(activity):
    transaction_type = activity.get("type", "").lower()
    balance_before = activity.get("balance_before")
    amount = activity.get("amount")
    if any(word in transaction_type for word in ["purchase", "withdrawal", "transfer"]):
        if amount > balance_before:
            granted = False
            balance_after = balance_before
            print(f"Insufficient funds for {transaction_type} in transaction. Granted set to False")
        else:
            balance_after = balance_before - amount
            granted = True
    else:
        balance_after = balance_before + amount
        granted = True
    activity["balance_after"] = balance_after
    activity["granted"] = granted
    return activity

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

def build_generation_prompt(strategy, global_clock, user_id, past_errors=None, fraud_label=0, balance_before=None, granted=None):
    """Builds the prompt for generating activity sequences, adapting based on fraud type and past errors."""

    # Adjust field list based on fraud or legit user
    if fraud_label == 0:
        # Legitimate transactions: LLM does NOT generate balance_after or granted
        fields_to_include = [
            "bank_timestamp", "local_timestamp", "account_id", "type", "amount",
            "currency", "balance_before", "location", "ip_address", "device_id",
            "network_type", "merchant_name", "recipient_id", "recipient_bank",
            "login_attempts", "session_id", "velocity", "distance_from_last_location", "is_repeat_location"
        ]
    else:
        # Fraudulent transactions: LLM must generate balance_after & granted
        fields_to_include = [
            "bank_timestamp", "local_timestamp", "account_id", "type", "amount",
            "currency", "balance_before", "balance_after", "granted", "location",
            "ip_address", "device_id", "network_type", "merchant_name", "recipient_id",
            "recipient_bank", "login_attempts", "session_id", "velocity",
            "distance_from_last_location", "is_repeat_location"
        ]

    # Generate a structured explanation of required fields
    field_explanation = "\n".join(
        [f"- `{field}`: {str(EXPECTED_FIELD_TYPES[field])}" for field in fields_to_include]
    )

    # Adjust JSON template example
    json_template = f"""
    ```json
    [
        {{
            "bank_timestamp": "2025-03-01T07:00:00+00:00",
            "local_timestamp": "2025-03-01T02:00:00-05:00",
            "account_id": "ACC-12345678",
            "type": "Purchase",
            "amount": 150.75,
            "currency": "USD",
            "balance_before": 1000.00,
            {'' if fraud_label == 0 else '"balance_after": 849.25,'}
            {'' if fraud_label == 0 else '"granted": true,'}
            "location": "New York, USA",
            "ip_address": "192.168.1.10",
            "device_id": "iPhone-14",
            "network_type": "Wi-Fi",
            "merchant_name": "Amazon",
            "recipient_id": null,
            "recipient_bank": null,
            "login_attempts": 1,
            "session_id": "S-12345678",
            "velocity": 0.52,
            "distance_from_last_location": 3.4,
            "is_repeat_location": true
        }}
    ]
    ```
    """

    return (
        f"You are an AI generating **detailed sequences of banking activities** for a fraud simulation.\n"
        f"Your task is to create a sequence of financial activities based on the predefined strategy below.\n\n"
        
        f"### Strategy:\n{strategy}\n\n"

        f"### Data Generation Rules:\n"
        f"- The financial activities must be generated as a JSON object.\n"
        f"- Only include the following fields:\n{field_explanation}\n\n"

        f"{'Fraudulent Transactions: Each new transaction should adapt to the previous balance_after value.' if fraud_label == 1 else ''}\n"
        
        f"{'Fraudulent Transactions: If the last transaction was denied (granted=False), adjust the fraud strategy.' if fraud_label == 1 and granted == False else ''}\n"

        f"- Ensure transactions follow **logical constraints**, such as:\n"
        f"  - Withdrawals or purchases should not exceed `balance_before`.\n"
        f"  - Transfers should respect available balance unless flagged as fraud.\n"

        f"- First, explain your reasoning **without making examples**.\n"
        f"- Then, generate **only** the structured JSON activity sequence as in the following JSON example:\n{json_template}\n"

        f"- The JSON MUST be enclosed within **triple backticks** using the format ```json ... ```.\n"
        f"- Ensure that timestamps are logically consistent and formatted according to ISO 8601 standards.\n"
        f"- Do **NOT** include comments or explanations in the JSON.\n"
        f"- End your response immediately after closing the JSON block.\n"
        
        f"### Important:\n"
        f"- **The JSON must be flat**\n"
        f"- Ensure all field values match the expected data types.\n"
        f"- **Errors you did in past generations and must avoid:** {past_errors}\n"
    )

#Add usefyul prints during build_generation_prompt
#- need to understand if the errors are beinng used and in case of invali structure it must be return retry


def generate_activity_sequence(strategy, global_clock=None, user_id=None, fraud_label=0):
    """Generates structured financial activities for an agent, handling both legitimate and fraudulent behaviors."""

    if fraud_label == 0:
        # ✅ Legitimate Users → Generate full sequence in one LLM call
        past_errors = read_past_errors()
        print(f"Generating full sequence for legitimate user {user_id}...")
        
        prompt = build_generation_prompt(strategy, global_clock, user_id, past_errors, fraud_label=0)
        response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}])
        raw_response = response['message']['content'].strip()
        save_to_text(raw_response, user_id)

        activity_sequence = validate_json(raw_response, user_id)
        if activity_sequence == 'retry':
            print(f"Retrying legitimate sequence for {user_id}...")
            return generate_activity_sequence(strategy, global_clock, user_id, fraud_label)
        
        # Post-process: Compute `balance_after` and `granted`
        for activity in activity_sequence:
            activity = compute_balance(activity)

        return activity_sequence  

    else:
        # ⚠️ Fraudulent Users → Step-by-step generation
        print(f"Generating fraudulent transactions dynamically for {user_id}...")
        
        balance_before = random.uniform(500, 5000)  # Initialize with a random balance
        fraudulent_transactions = []
        granted = True  # Assume first transaction is granted

        for _ in range(random.randint(2, 5)):  # Generate 2-5 fraudulent attempts
            past_errors = read_past_errors()
            print(f"Attempting fraud transaction for {user_id} with balance {balance_before}...")

            # Generate prompt with current balance_before and last granted status
            prompt = build_generation_prompt(
                strategy, global_clock, user_id, past_errors, fraud_label=1, 
                balance_before=balance_before, granted=granted
            )
            response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}])
            raw_response = response['message']['content'].strip()
            save_to_text(raw_response, user_id)

            activity = validate_json(raw_response, user_id)
            if activity == 'retry':
                print(f"Retrying fraudulent transaction for {user_id}...")
                continue
            
            # Ensure activity is structured correctly
            activity[0]['balance_before'] = balance_before
            activity[0] = compute_balance(activity[0])  
            
            # Update balance and granted status
            balance_before = activity[0]['balance_after']
            granted = activity[0]['granted']

            fraudulent_transactions.append(activity[0])

            # Adapt fraud behavior if the transaction was denied
            if not granted:
                print(f"🚨 Fraudulent transaction denied. Adjusting strategy for {user_id}...")
                strategy = modify_strategy_based_on_rejection(strategy)

        return fraudulent_transactions

def modify_strategy_based_on_rejection(strategy):
    """Modifies the fraud strategy when a transaction is denied, making the fraudster adapt."""
    
    print("🔄 Adapting fraud strategy...")

    # Example modifications: 
    # - Reduce transaction amount if previous fraud attempt was too large
    # - Try a different transaction type (e.g., from withdrawal to transfer)
    # - Use a different device or IP to bypass detection

    if "high-value transactions" in strategy:
        strategy = strategy.replace("high-value transactions", "medium-value transactions")

    if "single large withdrawal" in strategy:
        strategy = strategy.replace("single large withdrawal", "multiple small withdrawals")

    if "same device" in strategy:
        strategy = strategy.replace("same device", "new device")

    print(f"🚀 Updated fraud strategy: {strategy}")
    return strategy

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
    activity = compute_balance(activity)
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