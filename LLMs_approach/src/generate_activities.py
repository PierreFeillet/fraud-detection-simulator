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
from utilities import generate_random_hash, format_timestamp

# LLM used for sequence generation
#activity_model = 'mistral'
activity_model_id=watsonx_helper.activity_gen_model_id
activity_model= activity_model_id.split("/")[1]
strategy_model_id=watsonx_helper.strategy_gen_model_id
strategy_model=strategy_model_id.split("/")[0]
print(f"Activities will be generated using model: {activity_model_id}")

# File paths
OUTPUT_DIR = os.getcwd()+"/outputs/"+activity_model
os.makedirs(OUTPUT_DIR, exist_ok=True)
ERROR_TRACKING_FILE = os.path.join(OUTPUT_DIR, f"error_tracking.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, f"json_errors.log")
REWARD_LOG_FILE = os.path.join(OUTPUT_DIR, f"reward_progress.csv")
LOG_TEXT_FILE = os.path.join(OUTPUT_DIR, f"llm_chain_of_thought.txt")
DATA_DIR = os.getcwd()+"/data/"+activity_model
os.makedirs(DATA_DIR, exist_ok=True)

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
    "activity_id",  # activity ID first
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


def validate_json(text, user_id, last_timestamp=None):
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

        # Check if bank_timestamp is strictly later than last_timestamp (if provided)
        if last_timestamp:
            try:
                new_bank_ts = isoparse(tx["bank_timestamp"])
                prev_ts = isoparse(last_timestamp)
                # Ensure both datetime objects are offset-aware. If tzinfo is None, assume UTC.
                if new_bank_ts.tzinfo is None:
                    new_bank_ts = new_bank_ts.replace(tzinfo=timezone.utc)
                if prev_ts.tzinfo is None:
                    prev_ts = prev_ts.replace(tzinfo=timezone.utc)
                if new_bank_ts <= prev_ts:
                    errors.append(
                        f"Error: bank_timestamp {tx['bank_timestamp']} is not strictly later than last_timestamp {last_timestamp}."
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

def update_balance(tx, current_balance):
    """
    Computes the new balance based on the activity type and amount.
    For a deposit, adds the amount.
    For a withdrawal, transfer, purchase, or sale, subtracts the amount.
    """
    tx_type = tx.get("type", "").lower()
    amount = tx.get("amount", 0)
    if any(word in tx_type for word in ["deposit", "contribution", "transfer in"]):
        return current_balance + amount
    elif any(word in tx_type for word in ["withdrawal", "transfer out", "purchase", "sale"]):
        new_balance = current_balance - amount
        if new_balance<0:
            return current_balance
        else:
            return new_balance
    else:
        return current_balance

def assign_actvity_fields(activity, user_id, behavior_type, fraud_label):
    """Assigns additional fields to the activity."""
    activity['user_id'] = user_id
    activity['activity_id'] = f"TXN-{generate_random_hash(10)}"
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

def build_generation_prompt(strategy, user_id, history, balance, past_errors=None):
    """
    Builds a refined prompt for generating the next banking activity.
    - Strictly follows the given strategy for transaction type, amount range, location, velocity, etc.
    - Ensures timestamps follow chronological order.
    - Generates only 'bank_timestamp' (local timestamp is computed separately).
    - Prevents common JSON errors using past error feedback.

    Returns:
        A structured prompt for LLM-based activity generation.
    """

    # 1️⃣ **History Summary (if available)**
    history_summary = ""
    if history:
        last_tx = history[-1]
        last_timestamp = last_tx["bank_timestamp"]
        history_summary = (
            f"User {user_id}'s last recorded transaction:\n{json.dumps(last_tx, indent=2)}\n\n"
            f"- The new transaction's `bank_timestamp` must be STRICTLY AFTER {last_timestamp} (at least 1 minute later).\n"
        )
    else:
        last_timestamp = datetime.now(timezone.utc).isoformat()
        history_summary = (
            f"No previous transactions found for user {user_id}. "
            f"Generate the first transaction with a `bank_timestamp` AFTER {last_timestamp}.\n"
        )

    # 2️⃣ **Example JSON (LLM must strictly follow this format)**
    json_template = """
<<<JSON>>>        
{
  "bank_timestamp": "2025-03-01T10:15:32+00:00",
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

    # 3️⃣ **Field Explanations (Enforces JSON Structure)**
    field_explanation = (
        "- `bank_timestamp`: ISO 8601 UTC timestamp (STRICTLY increasing).\n"
        "- `account_id`: Must follow the format `ACC-XXXXXXXX`.\n"
        "- `type`: Must be one of the allowed transaction types (`Purchase`, `Transfer IN`, `Transfer OUT`, etc.).\n"
        "- `amount`: Must be within the range specified in the strategy and LESS than the `balance_before`.\n"
        "- `balance_before`: Account balance before the transaction.\n"
        "- `location`: City, Country (MUST match the geographic focus in the strategy).\n"
        "- `ip_address`: Must correspond to the transaction location (e.g., US-based IPs for US locations).\n"
        "- `device_id`: Device model (if unknown, set as `Unknown Device`).\n"
        "- `network_type`: `Wi-Fi` or `Cellular` (if unknown, set as `Unknown Network`).\n"
        "- `merchant_name`: Required for `Purchase` and `Sale`, MUST be null for Transfers.\n"
        "- `recipient_id` & `recipient_bank`: Required for Transfers, MUST be null for other transactions.\n"
    )

    # 4️⃣ **Main Prompt Assembly**
    prompt_parts = [
        history_summary,
        f"\n### Strategy Guidelines:\n{json.dumps(strategy, indent=2)}\n\n",
        "**Strictly adhere to this strategy when choosing transaction type, amount range, location, and other fields.**\n",
        "### Expected JSON Output Format:\n",
        json_template,
        "### Field Requirements:\n",
        field_explanation,
        "### Additional Constraints:\n",
        "- Generate **ONLY the `bank_timestamp`** (local timestamp will be computed separately).\n",
        "- Ensure `bank_timestamp` is strictly increasing compared to the previous transaction.\n",
        "- Derive `ip_address` realistically from the transaction location. Examples:\n",
        "  - USA locations → US-based IPv4 ranges (73.x.x.x, 24.x.x.x).\n",
        "  - Europe locations → European IPv4 ranges (81.x.x.x, 217.x.x.x).\n",
        "  - China locations → Chinese IPv4 ranges (202.x.x.x, 223.x.x.x).\n",
        "- Ensure the generated JSON follows a flat structure with NO comments or extra text.\n",
        "- DO NOT include any explanations, calculations, or metadata—ONLY return the JSON within <<<JSON>>> and <<<END_JSON>>>.\n"
    ]

    # 5️⃣ **Error Handling: Prevent Past Mistakes**
    if past_errors:
        prompt_parts.append(f"- Avoid repeating previous errors: {past_errors}.\n")

    return "".join(prompt_parts)


def enforce_timestamp_order(tx, last_timestamp_str):
    """
    Checks if tx["bank_timestamp"] is strictly later than last_timestamp_str.
    If not, adjusts tx["bank_timestamp"] to be at least 1 minute later and updates tx["local_timestamp"]
    to preserve the original time difference.
    """
    try:
        last_ts = isoparse(last_timestamp_str)
        orig_bank = isoparse(tx["bank_timestamp"])
        orig_local = isoparse(tx["local_timestamp"])
    except Exception as e:
        print("Error parsing timestamps:", e)
        return tx

    # Compute the original difference between bank_timestamp and local_timestamp
    # (this difference may include timezone offsets)
    delta = orig_bank - orig_local  # timedelta

    if orig_bank <= last_ts:
        # Enforce a minimum gap of 1 minute
        new_bank = last_ts + timedelta(minutes=1)
        tx["bank_timestamp"] = new_bank.isoformat()
        # Adjust local_timestamp to preserve the original time difference
        new_local = new_bank - delta
        tx["local_timestamp"] = new_local.isoformat()
        print(f"Adjusted timestamps: new bank_timestamp set to {tx['bank_timestamp']}, new local_timestamp set to {tx['local_timestamp']}")
    return tx


def generate_activity_sequence(strategy, user_id, behavior_type, fraud_label, num_activities=5, user_accounts=None):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)  # Convert string to dictionary
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in strategy: {strategy}")
    
    if user_accounts is None:
        user_accounts = {f"ACC-{generate_random_hash()}": assign_initial_balance()}  

    history_by_account = {acc: [] for acc in user_accounts}
    activities = []

    for i in range(num_activities):
        account_id = random.choice(list(user_accounts.keys()))
        current_balance = user_accounts[account_id]
        history = history_by_account[account_id]

        last_tx = history[-1] if history else None
        tx_location = select_valid_location(strategy.get("geographic_focus", ["Domestic US"]))
        bank_timestamp, local_timestamp = generate_local_and_bank_timestamp(tx_location, last_tx, strategy)

        retries = 0
        max_retries = 3

        while retries < max_retries:
            retries += 1
            if i == 0 or fraud_label == 0:
                # First transaction or legitimate profile
                tx = generate_static_activity(strategy, user_id, account_id, current_balance, local_timestamp)
            else:
                # Fraudulent transactions predicted by LLM
                past_errors = read_past_errors()
                prompt = build_generation_prompt(strategy, bank_timestamp, user_id, history, current_balance, past_errors)
                raw_response = watsonx_chat(
                    prompt=prompt,
                    model_id=activity_model_id,
                    parameters=watsonx_helper.parameters_activity
                )
                save_to_text(raw_response, user_id)
                tx, errors = validate_json(raw_response, user_id, last_tx.get("bank_timestamp") if last_tx else bank_timestamp)

                if tx == 'retry':
                    update_reward_log(score=-1, user_id=user_id, reason="; ".join(errors))
                    continue  # Retry generation
                else:
                    update_reward_log(score=1, user_id=user_id, reason="Successful JSON")
                    new_account_id = tx.get("account_id", account_id)
                    if new_account_id not in user_accounts:
                        user_accounts[new_account_id] = assign_initial_balance()
                        history_by_account[new_account_id] = []
                    account_id = new_account_id

            # Update balance ensuring correctness
            tx["balance_before"] = user_accounts[account_id]
            tx["balance_after"] = update_balance_for_account(tx, user_accounts)

            # Assign metadata fields, timestamps, and fraud labels **AFTER** balance updates
            tx = assign_activity_fields(tx, user_id, behavior_type, fraud_label)

            # Append to histories
            history_by_account[account_id].append(tx)
            activities.append(tx)

            break  # Exit retry loop on successful generation
        else:
            print(f"Failed to generate activity after {max_retries} retries for user {user_id}.")

    return activities


import pandas as pd
import random
from datetime import datetime, timezone

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, DATA_FILE='output_data.csv'):
    """
    Generates a bank log with fraudulent and legitimate agents.
    
    - Balances fraud and legitimate transactions based on target_fraud_percentage.
    - Efficiently writes to a CSV in batches.
    """

    fraudulent_strategies = load_existing_strategies(f"strategies/fraud_strategies_{strategy_model}.json")
    legitimate_strategies = load_existing_strategies(f"strategies/legitimate_strategies_{strategy_model}.json")
    
    global_clock = datetime.now(timezone.utc).isoformat()  # Start the global clock
    
    header_written = False  
    buffer = []
    
    def flush_buffer():
        """Writes buffered activities to CSV in batches."""
        nonlocal header_written
        if buffer:
            df = pd.DataFrame(buffer)
            df = df[ORDERED_COLUMNS]  # Ensure columns are ordered correctly
            df.to_csv(DATA_FILE, mode='a', index=False, header=not header_written)
            header_written = True  
            buffer.clear()
    
    total_generated = 0  # Track all activities (fraudulent + legitimate)
    target_fraud = int(total_activities * target_fraud_percentage)

    def generate_agent_activities(agent_count, fraud_label, strategies, remaining_activities):
        """Handles activity generation for fraud/legit agents."""
        nonlocal total_generated, global_clock
        for _ in range(agent_count):
            if total_generated >= remaining_activities:
                break

            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(strategies.keys()))
            strategy = strategies[behavior_type]

            # Decide the number of activities for this agent
            num_act = random.randint(1, 6)
            num_act = min(num_act, remaining_activities - total_generated)  # Ensure we don't exceed target

            activities = generate_activity_sequence(
                strategy=strategy, 
                user_id=user_id, 
                behavior_type=behavior_type, 
                fraud_label=fraud_label, 
                num_activities=num_act
            )

            # Add generated activities to buffer
            buffer.extend(activities)
            total_generated += len(activities)

            # Update global clock based on the last transaction generated
            if activities:
                global_clock = activities[-1]["bank_timestamp"]

            if total_generated >= remaining_activities:
                break

    # --- Generate Fraudulent Activities ---
    generate_agent_activities(fraud_agents_count, fraud_label=1, strategies=fraudulent_strategies, remaining_activities=target_fraud)

    # --- Generate Legitimate Activities ---
    generate_agent_activities(legit_agents_count, fraud_label=0, strategies=legitimate_strategies, remaining_activities=total_activities)

    # Final buffer flush to ensure everything is saved
    flush_buffer()
    
    print(f"Activity generation complete. Data saved to {DATA_FILE}")
    return pd.read_csv(DATA_FILE)

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
    plt.plot(df['timestamp'], df['cumulative_reward'], marker='o', linestyle='-', label='Cumulative Reward')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward Progress Over Time')
    plt.xticks(rotation=45)
    plt.legend()
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
    plt.plot(df.index, df['cumulative_success'], marker='o', linestyle='-', label='Cumulative Valid JSONs') #markevery=2,
    plt.xlabel('Attempt Number')
    plt.ylabel('Cumulative Valid JSONs')
    plt.title('JSON Success Rate')
    plt.grid(True)
    plt.legend()
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
    parser.add_argument('--target_fraud_percentage', help='Fraud rate', type=float, default=0.5) # Default is High Risk
    # Main simulation entry point
    start_time = time.time()
    cfg = parser.parse_args()
    pprint(cfg)
    print(f"Simulation started at {datetime.now().isoformat()}")
    DATA_FILE = os.path.join(DATA_DIR, f'fraud_simulation_activities_{format_number(cfg.nb_activities)}.csv')
    initialize_logs(DATA_FILE)
    generate_activities(total_activities=cfg.nb_activities, target_fraud_percentage=cfg.target_fraud_percentage, fraud_agents_count=cfg.fraud_agents_count, legit_agents_count=cfg.legit_agents_count, DATA_FILE=DATA_FILE)
    time_taken = round((time.time()-start_time)/60, 2)
    print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")
    visualize_json_success_rate()
    #visualize_rewards()