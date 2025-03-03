from collections import defaultdict
from datetime import datetime, timezone
import time
import json
import os
import random
import re
import uuid
from IPython import embed
import pandas as pd
import ollama  # or use watsonx_chat if needed
import matplotlib.pyplot as plt
import secrets

from watsonx_helper import watsonx_chat

# LLM used for sequence generation
LLM_model = 'mistral'
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
    "recipient_id": (str, type(None)),     # Can be string or null
    "recipient_bank": (str, type(None)),   # Can be string or null
    "login_attempts": int,
    "session_id": str,
    #"velocity": (int, float),
    #"distance_from_last_location": (int, float),
    "is_repeat_location": bool,
}

ORDERED_COLUMNS = [
    "transaction_id",  # Transaction ID first
    "user_id",
    # Time-related fields
    "bank_timestamp",
    "local_timestamp",
    # "velocity",
    # Geographical information
    "location",
    "ip_address",
    "device_id",
    "network_type",
    # Transaction fields
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
    # Behavior and fraud info
    "behavior_type",
    "fraud_label"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def initialize_logs():
    """Initializes all log files and clears the CSV."""
    for file in [REWARD_LOG_FILE, ERROR_LOG_FILE, LOG_TEXT_FILE, DATA_FILE, VALIDATION_LOG_FILE]:
        with open(file, 'w') as f:
            f.write("")
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    with open(DATA_FILE, 'w') as f:
        f.write("")  # Create an empty CSV file.

initialize_logs()

def log_error_occurrences(errors):
    """Logs and tracks how often each type of error occurs over time."""
    if os.path.exists(ERROR_TRACKING_FILE):
        with open(ERROR_TRACKING_FILE, "r", encoding="utf-8") as f:
            try:
                error_data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Corrupt or empty error tracking file. Resetting.")
                error_data = {"error_counts": defaultdict(int), "timestamps": []}
    else:
        error_data = {"error_counts": defaultdict(int), "timestamps": []}

    for error in errors:
        error_data["error_counts"][error] = error_data["error_counts"].get(error, 0) + 1

    error_data["timestamps"].append(datetime.now().isoformat())

    with open(ERROR_TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(error_data, f)

def log_json_validation(status, json_content, errors=None):
    """Logs JSON validation status and errors to a dedicated file."""
    log_file = os.path.join(OUTPUT_DIR, "json_validation.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- JSON Validation Result ---\n")
        f.write(f"Status: {status}\n")
        f.write(f"JSON:\n{json_content}\n")
        if errors:
            f.write("Errors:\n")
            for error in errors:
                f.write(f"- {error}\n")
        f.write("\n" + "="*80 + "\n")

def validate_json(text, user_id):
    """Validates JSON correctness and returns a list containing a single transaction.
    Returns 'retry' if the JSON is invalid."""
    errors = []
    print(f"Validating JSON for user {user_id}...")
    pattern_custom = r"<<<JSON>>>(.*?)<<<END_JSON>>>"
    pattern_code = r"```json\s*(.*?)\s*```"
    
    match = re.search(pattern_custom, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(pattern_code, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            errors.append("Error: JSON not enclosed in expected delimiters.")
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry'

    if re.search(r'//', json_str) or re.search(r'/\*.*?\*/', json_str, re.DOTALL):
        errors.append("Error: JSON contains inline comments.")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}: {errors}")
        return 'retry'

    try:
        activity = json.loads(json_str)
        if isinstance(activity, list):
            tx = activity[0]
        elif isinstance(activity, dict):
            tx = activity
            activity = [tx]
        else:
            errors.append("Error: JSON root is neither a list nor a dictionary.")
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry'
        
        missing_fields = [field for field in EXPECTED_FIELD_TYPES if field not in tx]
        extra_fields = [field for field in tx if field not in EXPECTED_FIELD_TYPES]
        if missing_fields:
            errors.append(f"Error: Missing fields: {', '.join(missing_fields)}")
        if extra_fields:
            errors.append(f"Error: Unexpected fields: {', '.join(extra_fields)}")
        if errors:
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry'

        for field, expected_type in EXPECTED_FIELD_TYPES.items():
            if field in tx and not isinstance(tx[field], expected_type):
                errors.append(f"Error: Field {field} expected type {expected_type} but got {type(tx[field])}")
        if errors:
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry'
        print(f"JSON validated successfully for user {user_id}.")
        return activity
    except json.JSONDecodeError as e:
        errors.append(f"JSON Decode Error: {e}")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}: {errors}")
        return 'retry'

def log_json_errors(error_array):
    """Logs an array of errors to a JSON file, one per line."""
    with open(ERROR_LOG_FILE, "a") as f:
        json.dump(error_array, f)
        f.write("\n")

def update_balance(tx, current_balance):
    """
    Computes the new balance based on the transaction type and amount.
    For a deposit, adds the amount.
    For a withdrawal, transfer, purchase, or sale, subtracts the amount.
    """
    tx_type = tx.get("type", "").lower()
    amount = tx.get("amount", 0)
    if "deposit" in tx_type:
        return current_balance + amount
    elif "withdrawal" or "transfer" or  "purchase" or "sale" in tx_type:
        new_balance = current_balance - amount
        if new_balance<0:
            return current_balance
        else:
            return new_balance
    else:
        return current_balance

def assign_actvity_fields(activity, user_id, behavior_type, fraud_label):
    """Assigns additional fields to the transaction."""
    activity['user_id'] = user_id
    activity['transaction_id'] = f"TXN-{generate_random_hash(10)}"
    activity['behavior_type'] = behavior_type
    activity['fraud_label'] = fraud_label
    return activity

def generate_random_hash(length=8):
    """Generates a random hexadecimal string of the given length."""
    return secrets.token_hex(length // 2)

def read_past_errors():
    """Reads past errors from the log file and returns them as a combined list."""
    if os.path.exists(ERROR_LOG_FILE):
        with open(ERROR_LOG_FILE, 'r') as file:
            try:
                lines = file.readlines()
                all_errors = []
                for line in lines:
                    try:
                        errors = json.loads(line.strip())
                        all_errors.extend(errors)
                    except json.JSONDecodeError:
                        continue
                return list(set(all_errors))
            except Exception as e:
                print(f"⚠️ Error reading past errors: {e}")
                return []
    return []

def build_generation_prompt(strategy, global_clock, user_id, history, balance, past_errors=None):
    """
    Builds a prompt for generating the next transaction.
    It includes a summary of previous transactions.
    """
    history_summary = ""
    if history:
        history_summary = f"Previous transactions for user {user_id}:\n" + json.dumps(history, indent=2) + "\n"

    json_template = """
<<<JSON>>>        
{
    "bank_timestamp": "2025-03-01T07:00:00+00:00",
    "local_timestamp": "2025-03-01T02:00:00-05:00",
    "account_id": "ACC-12345678",
    "type": "Purchase",
    "amount": 150.75,
    "currency": "USD",
    "balance_before": 1000.00,
    "location": "New York, USA",
    "ip_address": "192.168.1.10",
    "device_id": "iPhone-14",
    "network_type": "Wi-Fi",
    "merchant_name": "Amazon",
    "recipient_id": null,
    "recipient_bank": null,
    "login_attempts": 1,
    "session_id": "S-12345678",
    "is_repeat_location": true
}
<<<END_JSON>>>
"""
    field_explanation = (
        "- bank_timestamp: string (ISO 8601), the UTC time of the activity.\n"
        "- local_timestamp: string (ISO 8601), the local time with time zone offset.\n"
        "- account_id: string, formatted as \"ACC-XXXXXXXX\".\n"
        "- type: string, activity type.\n"
        "- amount: float, the monetary amount (0 if not applicable).\n"
        "- currency: string, e.g., \"USD\".\n"
        "- balance_before: float, the balance before the activity.\n"
        "- location: string, city and country.\n"
        "- ip_address: string, an IPv4 address.\n"
        "- device_id: string, the device model.\n"
        "- network_type: string, e.g., \"Wi-Fi\".\n"
        "- merchant_name: string or null.\n"
        "- recipient_id: string or null.\n"
        "- recipient_bank: string or null.\n"
        "- login_attempts: int, number of login attempts.\n"
        "- session_id: string, formatted as \"S-XXXXXXXX\".\n"
        "- is_repeat_location: bool, whether the location was used before.\n"
    )

    prompt_parts = []
    if history_summary:
        # If history exists, instruct the LLM to use the last bank_timestamp as reference,
        # and generate a new bank_timestamp that is a plausible continuation.
        prompt_parts.append(f"{history_summary}\n")
        prompt_parts.append("- Use the last bank_timestamp from the history as a reference to generate the next bank_timestamp. The new timestamp should be a plausible, later time, consistent with the transaction type and strategy.\n")
    else:
        # If no history, instruct the LLM to use the global clock as the starting point.
        prompt_parts.append(f"- Use the global clock {global_clock} as a reference to initialize the bank_timestamp for the activity. The generated timestamp should be close to this time.\n")
    prompt_parts.append(f"You are generating the next most probable banking activity for user {user_id} whose behavior is described by the following strategy: {strategy}.\n")
    prompt_parts.append(f"The activity must be generated in JSON format with the same fields as in the following example:\n{json_template}\n")
    prompt_parts.append("Do not include any extra fields, comments, or explanations.\n")
    prompt_parts.append("Only generate one transaction in this call.\n")
    prompt_parts.append("### Data Generation Rules:\n")
    prompt_parts.append(f"- Required fields and formats:\n{field_explanation}\n\n")
    prompt_parts.append("- Do not repeat the example; generate a new transaction.\n")
    prompt_parts.append("- The JSON MUST be enclosed within the markers <<<JSON>>> and <<<END_JSON>>>.\n")
    prompt_parts.append("- Ensure timestamps are ISO 8601 formatted and logically consistent.\n")
    prompt_parts.append("- End your response immediately after closing the JSON block.\n")
    prompt_parts.append("- The possible activity types are described in the strategy.\n")
    prompt_parts.append(f"- The value for `amount` must be <{balance}.\n")
    prompt_parts.append("### Important:\n")
    prompt_parts.append("- The JSON must be flat.\n")
    prompt_parts.append("- All string values must be enclosed in double quotes.\n")
    if past_errors:
        prompt_parts.append(f"- Do not repeat previous errors: {past_errors}.\n")
    
    return "".join(prompt_parts)

def generate_activity_sequence(strategy, global_clock, user_id, behavior_type, fraud_label, num_activities=5, user_accounts=None):
    """
    Iteratively generates a sequence of transactions for a user who may have multiple accounts.
    
    Returns a list of transactions.
    """
    if user_accounts is None: # Assign default values
        user_accounts = {f"ACC-{generate_random_hash()}": random.randint(100, 10000)}
    history_by_account = {acc: [] for acc in user_accounts}
    
    activities = []
    for i in range(num_activities):
        # Randomly select an account (LLM may output a new one)
        account_id = random.choice(list(user_accounts.keys()))
        current_balance = user_accounts[account_id]
        history = history_by_account.get(account_id, [])
        
        retries = 1
        while True:
            print(f"User {user_id}, Account {account_id}: Generating activity attempt {retries} with balance {current_balance:.2f}...")
            print(f"Generating activity {i+1}/{num_activities}...")
            past_errors = read_past_errors()
            prompt = build_generation_prompt(strategy, global_clock, user_id, history, current_balance, past_errors)
            response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}])
            raw_response = response['message']['content'].strip()
            save_to_text(raw_response, user_id)  # For debugging
            
            activity = validate_json(raw_response, user_id)
            if activity != 'retry':
                tx = activity[0]
                new_account_id = tx.get("account_id")
                if not new_account_id or new_account_id.strip() == "":
                    new_account_id = account_id
                    tx["account_id"] = new_account_id
                if new_account_id not in user_accounts:
                    default_balance = random.randint(100, 10000)
                    print(f"New account detected: {new_account_id}. Creating account with initial balance {default_balance:.2f}.")
                    user_accounts[new_account_id] = default_balance
                    history_by_account[new_account_id] = []
                current_balance = user_accounts[new_account_id]
                tx["balance_before"] = current_balance
                new_balance = update_balance(tx, current_balance)
                tx["balance_after"] = new_balance
                tx = assign_actvity_fields(tx, user_id, behavior_type, fraud_label)
                # Optionally, you could set a 'granted' field here if your strategy requires it.
                history_by_account[new_account_id].append(tx)
                user_accounts[new_account_id] = new_balance
                activities.append(tx)
                print(f"Activity generated for account {new_account_id}. New balance: {new_balance:.2f}")
                break
            else:
                print(f"Retrying activity generation for user {user_id}, account {account_id}...")
                update_reward_log(score=-1, user_id=user_id, reason="JSON generation failed")
                retries += 1
    return activities

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, buffer_size=1):
    """
    Generates a bank log with multiple fraudulent and legitimate agents.
    
    Creates transactions for each agent and writes them to a CSV.
    """
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
            df.to_csv(DATA_FILE, mode='a', index=False, header=not header_written)
            header_written = True  
            buffer.clear()
    
    total_generated = 0  # Count all transactions (fraudulent + legitimate)
    
    # Generate fraudulent activities.
    for _ in range(fraud_agents_count):
        user_id = f"USER-{generate_random_hash(8)}"
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]
        activities = generate_activity_sequence(strategy=strategy, global_clock=global_clock, user_id=user_id, fraud_label=1, behavior_type=behavior_type, num_activities=random.randint(1,6))
        for activity in activities: 
            buffer.append(activity)
            total_generated += 1
            if total_generated > int(total_activities * target_fraud_percentage):
                break
            if len(buffer) >= buffer_size:
                flush_buffer()
            
       
    # Generate legitimate activities for the remaining transactions.
    while total_generated <= total_activities:
        for _ in range(legit_agents_count):
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]
            activities = generate_activity_sequence(strategy=strategy, global_clock=global_clock, user_id=user_id, fraud_label=0, behavior_type=behavior_type, num_activities=random.randint(1,6))
            for activity in activities:
                buffer.append(activity)
                total_generated += 1
                if total_generated >= total_activities:
                    break
                if len(buffer) >= buffer_size:
                    flush_buffer() 
    print(f"Activity generation complete. Data saved to {DATA_FILE}")
    final_df = pd.read_csv(DATA_FILE)
    return final_df

def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

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
    with open(LOG_TEXT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### LLM Chain of Thought for user ID {agent_id} ###\n\n{reasoning_text}\n\n")

def update_reward_log(score: int, user_id: str, reason: str):
    with open(REWARD_LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()},{user_id},{score},{reason}\n")

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

# Main simulation entry point
start_time = time.time()
print(f"Simulation started at {datetime.now().isoformat()}")
print(DATA_FILE)

generate_activities(total_activities=20, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)
time_taken = round((time.time()-start_time)/60, 2)
print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")
visualize_json_success_rate()
visualize_rewards()