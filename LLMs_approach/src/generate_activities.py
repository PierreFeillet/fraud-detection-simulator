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
import ollama  # or use watsonx_chat if needed
import matplotlib.pyplot as plt
import secrets
from dateutil.parser import isoparse  # Install python-dateutil if needed
import csv
import argparse
from pprint import pprint
from watsonx_helper import watsonx_chat
import watsonx_helper
from static_behavior import generate_static_activity, assign_activity_fields, assign_initial_balance, select_valid_location, generate_local_and_bank_timestamp
from utilities import generate_random_hash, update_balance
# LLM used for sequence generation
#activity_model = 'mistral'


# Expected field types for the JSON schema
EXPECTED_FIELD_TYPES = {
    "bank_timestamp": str,  # ISO 8601 format
    "local_timestamp": str,  # ISO 8601 format
    "account_id": str,
    "type": str,
    "amount": (int, float),  # Allow both int and float for amounts
    "balance_before": (int, float),
    "location": str,
    "ip_address": str,
    "device_id": str,
    "network_type": str,
    "merchant_name": (str, type(None)),  # Can be string or null
    "recipient_id": (str, type(None)),     # Can be string or null
    "recipient_bank": (str, type(None)),   # Can be string or null
    #"login_attempts": int,
    #"session_id": str,
    #"velocity": (int, float),
    #"distance_from_last_location": (int, float),
    #"is_repeat_location": bool,
}

ORDERED_COLUMNS = [
    "transaction_id",  # transaction ID first
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
    # activity fields
    "type",
    "amount",
    "account_id",
    "balance_before",
    "granted",
    "balance_after",
    # Recipient details
    "merchant_name",
    "recipient_id",
    "recipient_bank",
    # Behavior and fraud info
    "behavior_type",
    "fraud_label"
]

def robust_remove(file_path, max_retries=5, delay=1):
    """Attempts to remove file_path, retrying if it fails."""
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            print(f"Attempt {i+1}: Failed to remove {file_path}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def initialize_logs(DATA_FILE):
    """Initializes all log files and clears the CSV.
    Writes a header to the reward log file."""
    # Define header for the reward log file.
    reward_header = "timestamp,user_id,reward,reason\n"
    
    # Write the header to the reward log file.
    with open(REWARD_LOG_FILE, 'w') as f:
        f.write(reward_header)
    
    # Clear the other log files.
    for file in [ERROR_LOG_FILE, LOG_TEXT_FILE, ]:
        with open(file, 'w') as f:
            f.write("")
    
    # Clear the CSV file.
    #if os.path.exists(DATA_FILE):
    #    os.remove(DATA_FILE)
    robust_remove(DATA_FILE)
    with open(DATA_FILE, 'w') as f:
        f.write("")  # Create an empty CSV file.

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

def extract_json_fragment(text):
    """
    Attempts to extract a JSON block from text using several common delimiter patterns.
    Returns the JSON string if found, otherwise None.
    """
    stripped = text.strip()
    # If the entire text is a JSON block (starts with '{' and ends with '}')
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    
    # First try your preferred delimiters
    match = re.search(r"<<<JSON>>>(.*?)<<<END_JSON>>>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Next, try markdown code block with "json" language tag
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Then, try any triple-backticks (without a language tag)
    match = re.search(r"```(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # As a last resort, try any block delimited by <<< and >>>
    match = re.search(r"<<<(.*?)>>>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None


def validate_json(text, user_id, last_tx_timestamp):
    """
    Validates JSON correctness and returns a list containing a single activity.
    Returns 'retry' if the JSON is invalid or if the new bank_timestamp is not strictly later
    than the provided last_timestamp.
    """
    errors = []
    print(f"Validating JSON for user {user_id}...")
    json_str = extract_json_fragment(text)
    if not json_str:
        errors.append("Error: JSON not enclosed in expected delimiters.")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}: {errors}")
        return 'retry', errors

    if re.search(r'//', json_str) or re.search(r'/\*.*?\*/', json_str, re.DOTALL):
        errors.append("Error: JSON contains inline comments.")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}: {errors}")
        return 'retry', errors

    try:
        activity = json.loads(json_str)
        # Standardize: we want exactly one activity dictionary.
        if isinstance(activity, list):
            if not activity:
                errors.append("Error: JSON list is empty.")
                log_json_errors(errors)
                return 'retry', errors
            tx = activity[0]
        elif isinstance(activity, dict):
            tx = activity
            activity = [tx]
        else:
            errors.append("Error: JSON root is neither a list nor a dictionary.")
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry', errors

        # Check if bank_timestamp is strictly later than last_timestamp (if prov ded)
        try:
            new_bank_ts = isoparse(tx["bank_timestamp"])
            prev_ts = isoparse(last_tx_timestamp)
            # Ensure both datetime objects are offset-aware. If tzinfo is None, assume UTC.
            if new_bank_ts.tzinfo is None:
                new_bank_ts = new_bank_ts.replace(tzinfo=timezone.utc)
            if prev_ts.tzinfo is None:
                prev_ts = prev_ts.replace(tzinfo=timezone.utc)
            if new_bank_ts <= prev_ts:
                errors.append(
                    f"Error: bank_timestamp {tx['bank_timestamp']} is not strictly later than last_timestamp {last_tx_timestamp}."
                )
                log_json_errors(errors)
                print(f"Errors detected in JSON for user {user_id}: {errors}")
                return 'retry', errors
        except Exception as e:
            errors.append(f"Error parsing timestamps: {e}")
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry', errors

        # Check for missing and extra fields
        missing_fields = [field for field in EXPECTED_FIELD_TYPES if field not in tx]
        extra_fields = [field for field in tx if field not in EXPECTED_FIELD_TYPES]
        if missing_fields:
            if "merchant_name" in missing_fields:
                errors.append(f"The activity was invalid because 'merchant_name' was missing for {tx.get('type')}. Generate a valid JSON activity with the required merchant_name.")
            elif any(word in missing_fields for word in ["recipient_id", "recipient_bank"]):
                errors.append(f"The activity was invalid because 'recipient_id' and 'recipient_bank' were missing for {tx.get('type')}. Generate a valid JSON activity with the required fields.")
            else:
                errors.append(f"Error: Missing fields: {', '.join(missing_fields)}")
        if extra_fields:
            errors.append(f"Error: Unexpected fields: {', '.join(extra_fields)}")
        if errors:
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry', errors

        # Check for data type consistency
        for field, expected_type in EXPECTED_FIELD_TYPES.items():
            if field in tx and not isinstance(tx[field], expected_type):
                errors.append(f"Error: Field {field} expected type {expected_type} but got {type(tx[field])}")
        if errors:
            log_json_errors(errors)
            print(f"Errors detected in JSON for user {user_id}: {errors}")
            return 'retry', errors

        print(f"JSON validated successfully for user {user_id}.")
        return activity, None

    except json.JSONDecodeError as e:
        errors.append(f"JSON Decode Error: {e}")
        log_json_errors(errors)
        print(f"Errors detected in JSON for user {user_id}: {errors}")
        return 'retry', errors


def log_json_errors(error_array):
    """Logs an array of errors to a JSON file, one per line."""
    with open(ERROR_LOG_FILE, "a") as f:
        json.dump(error_array, f)
        f.write("\n")

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

def build_generation_prompt(strategy, user_id, history, accounts, past_errors=None):
    """
    Builds a refined prompt for generating the next banking activity.
    - Strictly follows the given strategy for transaction type, amount range, location, velocity, etc.
    - Ensures timestamps follow chronological order.
    - Generates only 'bank_timestamp' (local timestamp is computed separately).
    - Prevents common JSON errors using past error feedback.

    Returns:
        A structured prompt for LLM-based activity generation.
    """
    last_tx = history[-1]
    last_timestamp = last_tx["bank_timestamp"]
    json_template = """
<<<JSON>>>        
{
  "bank_timestamp": "2025-03-01T10:15:32+00:00",
  "local_timestamp": "2025-03-01T05:15:32-05:00",
  "account_id": "ACC-82736401",
  "type": "Purchase",
  "amount": 45.99,
  "balance_before": 1280.45,
  "location": "Chicago, USA",
  "ip_address": "73.56.201.89",
  "device_id": "iPhone-13",
  "network_type": "Wi-Fi",
  "merchant_name": "Starbucks",
  "recipient_id": null,
  "recipient_bank": null
}
<<<END_JSON>>>
"""

    #  **Field Explanations (Enforces JSON Structure)**
    field_explanation = (
        "- `bank_timestamp`: ISO 8601 UTC timestamp (STRICTLY increasing).\n"
        "-`local_timestamp`: ISO 8601 is the bank_timestamp converted in the `location` timezone.\n"
        "- `account_id`: Must follow the format `ACC-XXXXXXXX`. This is the account the user operates on for the generated transaction.\n"
        "- `type`: Must be one of the allowed transaction types (`Purchase`, `Transfer IN`, `Transfer OUT`, etc.).\n"
        f"- `amount`: Must be within the range specified in the strategy .\n"
        "- `balance_before`: Account balance before the transaction.\n"
        "- `location`: City, Country (MUST match the geographic focus in the strategy).\n"
        "- `ip_address`: Must correspond to the transaction location (e.g., US-based IPs for US locations).\n"
        "- `device_id`: Device model (if unknown, set as `Unknown Device`).\n"
        "- `network_type`: `Wi-Fi` or `Cellular` (if unknown, set as `Unknown Network`).\n"
        "- `merchant_name`: Required for purchases and salea, MUST be null for Transfers.\n"
        "- `recipient_id` & `recipient_bank`: Required for Transfers, MUST be null for other transactions.\n"
    )

    # **Main Prompt Assembly**
    prompt_parts = [
        f"Your task is to generate the next most probable bank activity for user {user_id} whose profile is described by the following strategy:\n",
        f"\n### Strategy Guidelines:\n{json.dumps(strategy, indent=2)}\n\n",
        f"The most probable transaction should follow the user's historical behavior and the strategy guidelines.\n",
        f"### History Summary:\n{history}\n",
        "**Strictly adhere to this strategy when choosing transaction type, amount range, location, and other fields.**\n",
        "The generated tarnsaction and its charactheristics must be returned as a JSON file in the delimiters <<<START>>> ... <<<END_JSON>>>\n",
        "### Example JSON:\n",
        json_template,
        "### Field Requirements:\n",
        field_explanation,
        "### Additional Constraints:\n",
        f"- The `amount` must be within the strategy's typical amount range and smaller than the latest balance on the chosen account.\n",
        f"The possible accounts the user can operate on are: {accounts}. Pick the most probable one.\n",
        "- **Realistic Time Differences:** Ensure that the time gap between the current transaction bank_timestamp and the previous transaction is realistic given the geographical locations. For instance, if the previous transaction was in Paris and the current one is in Berlin (or another distant European city), the time interval should be long enough to be plausible—if it’s too short, it would trigger suspicion.\n",
        f"- Ensure `bank_timestamp` is strictly increasing compared to the previous transaction bank_timestamp {last_timestamp}.\n",
        "- Derive `ip_address` realistically from the transaction location. Examples:\n",
        "  - USA locations → US-based IPv4 ranges (73.x.x.x, 24.x.x.x).\n",
        "  - Europe locations → European IPv4 ranges (81.x.x.x, 217.x.x.x).\n",
        "  - China locations → Chinese IPv4 ranges (202.x.x.x, 223.x.x.x).\n",
        "- Ensure the generated JSON follows a flat structure with NO comments or extra text.\n",
        "- DO NOT include any explanations, calculations, or metadata—ONLY return the JSON within <<<JSON>>> and <<<END_JSON>>>.\n"
    ]

    # **Error Handling: Prevent Past Mistakes**
    if past_errors:
        prompt_parts.append(f"- Avoid repeating these errors: {past_errors}.\n")

    return "".join(prompt_parts)


def generate_activity_sequence(strategy, user_id, behavior_type, fraud_label, global_clock, num_activities=5, user_accounts=None):
    if isinstance(strategy, str):
        try:
            match = re.search(r'{.*}', strategy, re.DOTALL)
            if match:
                json_str = match.group(0)
            strategy_json = json.loads(json_str)  # Convert string to dictionary
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in strategy: {strategy}")
    
    # Initialize as many accounts as in strategy and assign the initial balance
    n_accounts = strategy_json.get("n_accounts", 1)
    accounts = {f"ACC-{generate_random_hash()}": assign_initial_balance(strategy_json) for n in range(n_accounts)}  
    print("Initialized user accounts:", accounts)
    #history_by_account = {acc: [] for acc in accounts}
    activities = []

    for i in range(num_activities):
        retries = 0

        while True:
            retries += 1
            print(f"Generating activity {i+1}/{num_activities} for user {user_id}: attempt {retries}")
            if i == 0 or fraud_label == 0:
                print("Chosen apporach: statitic")
                # First transaction or legitimate profile
                # Pick account randomly and related transcation history
                account_id = random.choice(list(accounts.keys()))
                #history = history_by_account[account_id]
                last_tx = activities[-1] if activities else None
                tx = generate_static_activity(strategy_json, user_id, account_id, global_clock, last_tx)
                #embed() 
            else:
                print("Chosen approach: LLM")
                # Fraudulent transactions predicted by LLM
                past_errors = read_past_errors()
                # Feed LLM with all history for that user (to allow connections among accounts)
                prompt = build_generation_prompt(strategy, user_id, activities, accounts, past_errors)
                raw_response = watsonx_chat(
                    prompt=prompt,
                    model_id=activity_model_id,
                    parameters=watsonx_helper.parameters_activity
                )
                save_to_text(raw_response, user_id)
                tx, errors = validate_json(raw_response, user_id, activities[-1].get("bank_timestamp"))
                
                if tx == 'retry':
                    update_reward_log(score=-1, user_id=user_id, reason="; ".join(errors))
                    print(f"Activity generation failed for user {user_id} after {retries} retries.")
                    continue  # Retry generation
                else:
                    update_reward_log(score=1, user_id=user_id, reason="Successful JSON")
                    tx = tx[0]
                    new_account_id = tx.get("account_id", account_id)
                    # Chcek if account exists
                    if new_account_id not in accounts:
                        accounts[new_account_id] = assign_initial_balance(strategy_json)

            tx['amount'] = round(tx.get("amount", 0), 2)  # Round amount to 2 decimal places
            # Update balance ensuring correctness
            tx["balance_before"] = round(accounts[tx["account_id"]],2)
            tx["balance_after"] = update_balance(tx)
            # Update account balance
            accounts[tx["account_id"]] = tx["balance_after"]
            #tx["balance_after"] = update_balance_for_account(tx, user_accounts)

            # Assign metadata fields, timestamps, and fraud labels **AFTER** balance updates
            tx = assign_activity_fields(tx, user_id, behavior_type, fraud_label)
            # Enforce realistic timestamp ordering based on the previous transaction
            activities.append(tx)
            break  # Exit retry loop on successful generation
    print("------------------------------------------------")
    return activities

def flush_buffer(buffer, data_file, header_written):
    """
    Writes the current buffer to the CSV file and clears the buffer.
    Returns True indicating that the header is now written.
    """
    if buffer:
        df = pd.DataFrame(buffer)
        df = df[ORDERED_COLUMNS]  # ensure correct column order
        df.to_csv(data_file, mode='a', index=False, header=not header_written)
        buffer.clear()
        header_written = True
    return header_written

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, DATA_FILE='output_data.csv', buffer_size=2):
    """
    Generates a bank log with both fraudulent and legitimate agents.
    Fraudulent activities are generated first until the target fraud count is reached,
    then legitimate activities are generated until total_activities is met.
    Activities are written to CSV in batches based on the buffer_size.
    Finally, the CSV is sorted by bank_timestamp to interleave transactions realistically.
    """
    # Load strategies
    fraudulent_strategies = load_existing_strategies(f"strategies/fraud_strategies_{strategy_model}.json")
    legitimate_strategies = load_existing_strategies(f"strategies/legitimate_strategies_{strategy_model}.json")
    
    # Determine target counts
    target_fraud = int(total_activities * target_fraud_percentage)
    
    # Clear (or create) the CSV file by writing an empty DataFrame with header
    df_empty = pd.DataFrame(columns=ORDERED_COLUMNS)
    df_empty.to_csv(DATA_FILE, index=False)
    
    # We'll use a buffer to accumulate transactions before writing them out.
    buffer = []
    header_written = True  # Header was written above
    
    # Global clock 
    global_clock = datetime.now(timezone.utc).isoformat()

    # --- Generate Fraudulent Activities ---
    fraud_count = 0
    while fraud_count < target_fraud:
        user_id = f"USER-{generate_random_hash(8)}"
        behavior_type = random.choice(list(fraudulent_strategies.keys()))
        strategy = fraudulent_strategies[behavior_type]
        # Generate a random number of activities for this agent without exceeding the fraud target
        num_act = random.randint(1, 6)
        num_act = min(num_act, target_fraud - fraud_count)
        print("\n")
        activities = generate_activity_sequence(
            strategy=strategy, 
            user_id=user_id, 
            behavior_type=behavior_type, 
            fraud_label=1, 
            global_clock=global_clock,
            num_activities=num_act
        )
        buffer.extend(activities)
        fraud_count += len(activities)
        
        # Update global clock based on the last transaction if available
        #if activities:
        #    global_clock = activities[-1]["bank_timestamp"]
        
        # Flush the buffer if it has reached the desired size
        if len(buffer) >= buffer_size:
            header_written = flush_buffer(buffer, DATA_FILE, header_written)

    # --- Generate Legitimate Activities ---
    total_generated = fraud_count  # fraudulent activities already generated
    while total_generated < total_activities:
        user_id = f"USER-{generate_random_hash(8)}"
        behavior_type = random.choice(list(legitimate_strategies.keys()))
        strategy = legitimate_strategies[behavior_type]
        num_act = random.randint(1, 6)
        num_act = min(num_act, total_activities - total_generated)
        
        activities = generate_activity_sequence(
            strategy=strategy, 
            user_id=user_id, 
            behavior_type=behavior_type, 
            fraud_label=0, 
            global_clock=global_clock,
            num_activities=num_act
        )
        buffer.extend(activities)
        total_generated += len(activities)
        
        #if activities:
        #    global_clock = activities[-1]["bank_timestamp"]
        
        if len(buffer) >= buffer_size:
            header_written = flush_buffer(buffer, DATA_FILE, header_written)
    
    # Flush any remaining activities in the buffer
    if buffer:
        header_written = flush_buffer(buffer, DATA_FILE, header_written)
    
    # --- Merge and Sort for Realism ---
    # Read the CSV, sort by bank_timestamp, and write back the sorted CSV.
    df_final = pd.read_csv(DATA_FILE)
    #df_final.sort_values(by='bank_timestamp', key=lambda col: pd.to_datetime(df_final['bank_timestamp']), inplace=True)
    df_final.to_csv(DATA_FILE, index=False)
    
    print(f"Activity generation complete. Data saved to {DATA_FILE}")
    return df_final


def flush_buffer_immediate(tx, DATA_FILE):
    """Immediately appends a single activity to the CSV file."""
    df = pd.DataFrame([tx])
    df = df[ORDERED_COLUMNS]
    header = not (os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0)
    df.to_csv(DATA_FILE, mode='a', index=False, header=header)

def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_to_text(reasoning_text, agent_id=None):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(LOG_TEXT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### LLM Chain of Thought for user ID {agent_id} ###\n\n{reasoning_text}\n\n")

def update_reward_log(score: int, user_id: str, reason: str):
    with open(REWARD_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), user_id, score, reason])

def visualize_rewards():
    """Plots cumulative reward progress over time."""
    if not os.path.exists(REWARD_LOG_FILE):
        print("No reward log found.")
        return
    # Read the reward log, assuming it has a header.
    df = pd.read_csv(REWARD_LOG_FILE)
    # Convert timestamp to datetime.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Ensure rewards are numeric.
    df['reward'] = df['reward'].astype(int)
    # Sort by timestamp for a chronological plot.
    df.sort_values(by='timestamp', inplace=True)
    # Compute cumulative reward.
    df['cumulative_reward'] = df['reward'].cumsum()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['cumulative_reward'], marker='o', linestyle='-', label='Cumulative Reward',)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Reward', fontsize=16)
    plt.title('Reward Progress Over Time', fontsize=16)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reward_trend.png"))

def visualize_json_success_rate():
    """Plots the cumulative count of successful JSON generations over attempt number."""
    if not os.path.exists(REWARD_LOG_FILE):
        print("No reward log found.")
        return
    # Read the reward log. Adjust the names if needed.
    df = pd.read_csv(REWARD_LOG_FILE)
    # Convert rewards to numeric, if necessary.
    df['reward'] = df['reward'].astype(int)
    # Assume reward of 2 means success.
    df['success'] = df['reward'].apply(lambda r: 1 if r == 1 else 0)
    df['cumulative_success'] = df['success'].cumsum()
    
    plt.figure(figsize=(10, 10))
    plt.plot(df.index, df['cumulative_success'], marker='o', linestyle='-', label='Cumulative Valid JSONs',) #markevery=2,
    plt.xlabel('Attempt Number', fontsize=16)
    plt.ylabel('Cumulative Valid JSONs', fontsize=16)
    plt.title('JSON Success Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "success_rate.png"))
    print(f"Cumulative reward plot created at {OUTPUT_DIR} ")

def format_number(nb_global_activities):
    if nb_global_activities >= 1_000_000:
        value = nb_global_activities / 1_000_000
        suffix = "M"
    elif nb_global_activities >= 1_000:
        value = nb_global_activities / 1_000
        suffix = "K"
    else:
        return str(nb_global_activities)  # No suffix for numbers less than 1,000

    # Format to remove .0 if the value is an integer
    formated_number = f"{int(value) if value.is_integer() else round(value, 1)}{suffix}"
    return formated_number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nb_activities', help='Total number of activities to be generated', type=int, required=True)
    parser.add_argument('--fraud_agents_count', help='Number of fraudulent agents', type=int, default=4)
    parser.add_argument('--legit_agents_count', help='Number of legitimate agents', type=int, default=40)
    parser.add_argument('--target_fraud_percentage', help='Fraud rate', type=float, default=0.05) # Default is High Risk
    # Main simulation entry point
    start_time = time.time()
    cfg = parser.parse_args()
    pprint(cfg)
    print(f"Simulation started at {datetime.now().isoformat()}")
    activity_model_id=watsonx_helper.activity_gen_model_id
    activity_model= activity_model_id.split("/")[1]
    strategy_model_id=watsonx_helper.strategy_gen_model_id
    strategy_model=strategy_model_id.split("/")[0]
    print(f"Activities will be generated using model: {activity_model_id}")

    # File paths
    OUTPUT_DIR = os.getcwd()+"/outputs/"+activity_model+"/"+format_number(cfg.nb_activities)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ERROR_TRACKING_FILE = os.path.join(OUTPUT_DIR, f"error_tracking.json")
    ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_errors.log")
    REWARD_LOG_FILE = os.path.join(OUTPUT_DIR, f"reward_progress.csv")
    LOG_TEXT_FILE = os.path.join(OUTPUT_DIR, f"llm_chain_of_thought.txt")
    DATA_DIR = os.getcwd()+"/data/"+activity_model
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_FILE = os.path.join(DATA_DIR, f'fraud_simulation_activities_{format_number(cfg.nb_activities)}.csv')
    initialize_logs(DATA_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_activities(total_activities=cfg.nb_activities, target_fraud_percentage=cfg.target_fraud_percentage,  DATA_FILE=DATA_FILE)
    time_taken = round((time.time()-start_time)/60, 2)
    print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")
    visualize_json_success_rate()
    #visualize_rewards()