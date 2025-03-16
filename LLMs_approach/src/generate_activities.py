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
        return round(current_balance + amount,2)
    elif any(word in tx_type for word in ["withdrawal", "transfer out", "purchase", "sale"]):
        new_balance = round(current_balance - amount,2)
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

def build_generation_prompt(strategy, global_clock, user_id, history, balance, past_errors=None):
    """
    Builds a prompt for generating the next activity.
    It includes a summary of previous activities, plus explicit rules
    for time-series consistency and activity structure.
    """

    # 1. Summarize the user's history (if available).
    history_summary = ""
    if history:
        history_summary = f"Previous activities for user {user_id}:\n" + json.dumps(history, indent=2) + "\n"
        last_timestamp = history[-1].get("bank_timestamp", global_clock)
        time_instruction = (
            f"Use the last bank_timestamp {last_timestamp} as a reference for the next activity. "
            "The new bank_timestamp must be strictly later (chronologically) than the last one by at least 1 minute. "
            "If the new bank_timestamp is equal to or earlier than the last one, add sufficient delay. "
            "Generate a plausible gap in time based on the activity type and the user's typical activity pattern."
        )
    else:
        # If no history, we start from the global clock.
        time_instruction = (
            f"Use the global clock {global_clock} as a reference for the activity. "
            "The generated bank_timestamp should be close to this time."
        )

    # 2. Provide a JSON template as an example
    json_template = """
<<<JSON>>>        
{
  "bank_timestamp": "2025-03-01T10:15:32+00:00",
  "local_timestamp": "2025-03-01T05:15:32-05:00",
  "account_id": "ACC-82736401",
  "type": "Purchase",
  "balance_before": 1280.45,
  "amount": 45.99,
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

    # 3. Explain the fields
    field_explanation = (
        "- bank_timestamp: string (ISO 8601), the UTC time of the activity.\n"
        "- local_timestamp: string (ISO 8601), the local time with correct offset.\n"
        "- account_id: string, formatted as \"ACC-XXXXXXXX\".\n"
        "- type: string, activity type (Purchase, Sale, Transfer IN, Transfer Out, Withdrawal, etc.).\n"
        "- balance_before: float, the balance before the activity.\n"
        f"- amount: float, the monetary amount (0 if not applicable), must be <{balance}.\n"
        "- location: string, city and country.\n"
        "- ip_address: string, a valid IPv4 address.\n"
        "- device_id: string, the device model.\n"
        "- network_type: string, e.g., \"Wi-Fi\", \"Cellular\".\n"
        "- merchant_name: string (null if not applicable).\n"
        "- recipient_id: string formatted as \"REC-XXXXXXXX\" (null if not applicable).\n"
        "- recipient_bank: string (null if not applicable).\n"
    )

    # 4. Build the final prompt in parts
    prompt_parts = []

    # 4a. If we have prior activities, show them and add extra time instructions
    if history_summary:
        prompt_parts.append(history_summary)
        prompt_parts.append(
            f"- The new bank_timestamp must be strictly after {last_timestamp} (at least 1 minute later).\n"
            f"- Do not produce a timestamp equal to or earlier than {last_timestamp}. If such a time is produced, add additional delay.\n"
        )
    else:
        prompt_parts.append(
            f"- No prior history. Use the global clock {global_clock} as a reference, and generate a bank_timestamp that is strictly later than {global_clock} (by at least 1 minute).\n")

    # Add the primary time_instruction
    prompt_parts.append(time_instruction + "\n")

    # 4b. High-level generation instruction
    prompt_parts.append(
        f"You are generating the next probable banking activity for user {user_id} with strategy: {strategy}.\n"
    )
    prompt_parts.append(
        "The output must be valid JSON with **exactly** the same fields as in this example:\n"
        f"{json_template}\n"
    )
    prompt_parts.append("Do not include any extra fields, comments, or explanations.\n")
    prompt_parts.append("Only generate one activity in this call.\n")

    # 4c. Data Generation Rules:
    prompt_parts.append("### Data Generation Rules:\n")
    prompt_parts.append(f"- Required fields and formats:\n{field_explanation}\n\n")
    prompt_parts.append("- Use the strategy details above solely for internal reasoning. Do not include any part of these details in your output. Generate only the final activity JSON enclosed in <<<JSON>>> and <<<END_JSON>>>..\n")
    prompt_parts.append("- Keep timestamps logically consistent (UTC vs. local time offset).\n")
    prompt_parts.append("- The local_timestamp must match the time zone offset implied by the location.\n")
    prompt_parts.append("- The hour in local_timestamp cannot exceed 23.\n")
    prompt_parts.append("- End your response immediately after closing the JSON block.\n")

    # 4d. activity-type logic:
    prompt_parts.append("- For any Transfer activity, set 'type' to 'Transfer IN' or 'Transfer Out'. Provide recipient_id and recipient_bank, but merchant_name must be null.\n")
    prompt_parts.append("- For a Purchase or Sale, merchant_name must not be null, but recipient_id and recipient_bank must be null.\n")
    prompt_parts.append("- For a Withdrawal, deposit, or other, merchant_name and recipient_id can be null if not applicable.\n")

    # 4e. Additional constraints from user 
    prompt_parts.append("- IP addresses should be plausible (each octet 0–255). Avoid placeholders like 999.999.\n")
    prompt_parts.append("- If location is e.g. 'New York, USA', consider UTC-5 or UTC-4 (depending on date). If 'Shanghai, China', consider UTC+8.\n")
    prompt_parts.append("- Timestamps must strictly increase with each new activity for the same user.\n")
    prompt_parts.append("- It is very important that data are realistic!!\n")

    # 4f. Include past errors if provided
    if past_errors:
        prompt_parts.append(f"- Do not repeat previous errors: {past_errors}.\n")
    
    # Return the assembled prompt
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


def generate_activity_sequence(strategy, global_clock, user_id, behavior_type, fraud_label, num_activities=5, user_accounts=None):
    """
    Iteratively generates a sequence of activities for a user who may have multiple accounts.
    
    Returns a list of activities.
    """
    if user_accounts is None:  # Assign default values
        user_accounts = {f"ACC-{generate_random_hash()}": random.randint(100, 10000)}
    history_by_account = {acc: [] for acc in user_accounts}
    
    activities = []
    for i in range(num_activities):
        # Randomly select an account (LLM may output a new one)
        account_id = random.choice(list(user_accounts.keys()))
        # Get balance for that account
        current_balance = user_accounts[account_id]
        history = history_by_account.get(account_id, [])
        # Determine last timestamp from history, or use global_clock if none
        last_timestamp = history[-1].get("bank_timestamp", global_clock) if history else global_clock
        
        retries = 1
        while True:
            print(f"User {user_id}, Account {account_id}: Generating activity attempt {retries} with balance {current_balance:.2f}...")
            print(f"Generating activity {i+1}/{num_activities}...")
            past_errors = read_past_errors()
            #options = {"temperature": 0.8, "top_p": 0.9}
            #if retries > 1: # try changing parameter settings
            #    options = {"temperature": 0.5, "top_p": 0.8}
            #print(options)
            prompt = build_generation_prompt(strategy, global_clock, user_id, history, current_balance, past_errors)
            #response = ollama.chat(model=activity_model, messages=[{"role": "user", "content": prompt}], options=options)
            #raw_response = response['message']['content'].strip()
            raw_response=watsonx_chat(prompt=prompt, model_id=activity_model_id, parameters=watsonx_helper.parameters_activity)
            save_to_text(raw_response, user_id)  # For debugging
            
            # Pass the last_timestamp into validate_json to enforce timestamp order.
            activity, errors = validate_json(raw_response, user_id, last_timestamp)
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
                current_balance = round(user_accounts[new_account_id],2)
                tx["balance_before"] = current_balance
                new_balance = round(update_balance(tx, current_balance),2)
                tx["balance_after"] = new_balance
                tx = assign_actvity_fields(tx, user_id, behavior_type, fraud_label)
                history_by_account[new_account_id].append(tx)
                user_accounts[new_account_id] = new_balance
                tx['bank_timestamp'] = format_timestamp(tx['bank_timestamp'])
                tx['local_timestamp'] = format_timestamp(tx['local_timestamp'])
                activities.append(tx)
                print(f"Activity generated for account {new_account_id}. New balance: {new_balance:.2f}")
                update_reward_log(score=1, user_id=user_id, reason="JSON generation successful")
                break
            else:
                print(f"Retrying activity generation for user {user_id}, account {account_id}...")
                joined_errors = "; ".join(errors)
                update_reward_log(score=-1, user_id=user_id, reason=joined_errors)
                retries += 1
    return activities

def format_timestamp(time):
    parsed_time = isoparse(time)
    st_time = parsed_time.isoformat(timespec='seconds') 
    return st_time

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, DATA_FILE='output_data.csv'):
    """
    Generates a bank log with multiple fraudulent and legitimate agents.
    
    Creates activities for each agent and writes them to a CSV immediately.
    """
    fraudulent_strategies = load_existing_strategies(f"strategies/fraud_strategies_{strategy_model}.json")
    legitimate_strategies = load_existing_strategies(f"strategies/legitimate_strategies_{strategy_model}.json")
    
    global_clock = datetime.now(timezone.utc).isoformat()

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
    
    total_generated = 0  # Count all activities (fraudulent + legitimate)
    target_fraud = int(total_activities * target_fraud_percentage)
    
    # --- Fraudulent activities generation ---
    while total_generated < target_fraud:
        remaining = target_fraud - total_generated
        for _ in range(fraud_agents_count):
            if total_generated >= target_fraud:
                break
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(fraudulent_strategies.keys()))
            strategy = fraudulent_strategies[behavior_type]
            # Decide the number of activities for this agent, but don't generate more than needed.
            num_act = random.randint(1, 6)
            num_act = min(num_act, remaining)
            activities = generate_activity_sequence(
                strategy=strategy, 
                global_clock=global_clock, 
                user_id=user_id, 
                behavior_type=behavior_type, 
                fraud_label=1, 
                num_activities=num_act
            )
            for activity in activities:
                assign_actvity_fields(activity, user_id, behavior_type, fraud_label=1)
                flush_buffer_immediate(activity, DATA_FILE)  # Immediately flush this activity to CSV.
                total_generated += 1
                if total_generated >= target_fraud:
                    break
            if total_generated >= target_fraud:
                break

    # --- Legitimate activities generation ---
    while total_generated < total_activities:
        remaining = total_activities - total_generated
        for _ in range(legit_agents_count):
            if total_generated >= total_activities:
                break
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]
            num_act = random.randint(1, 6)
            num_act = min(num_act, remaining)
            activities = generate_activity_sequence(
                strategy=strategy, 
                global_clock=global_clock, 
                user_id=user_id, 
                behavior_type=behavior_type, 
                fraud_label=0, 
                num_activities=num_act
            )
            for activity in activities:
                assign_actvity_fields(activity, user_id, behavior_type, fraud_label=0)
                flush_buffer_immediate(activity, DATA_FILE)
                total_generated += 1
                if total_generated >= total_activities:
                    break
            if total_generated >= total_activities:
                break
        if total_generated >= total_activities:
            break

    # Final flush in case anything remains.
    if buffer:
        flush_buffer()
    
    print(f"Activity generation complete. Data saved to {DATA_FILE}")
    final_df = pd.read_csv(DATA_FILE)
    return final_df

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
    initialize_logs(DATA_FILE)
    generate_activities(total_activities=cfg.nb_activities, target_fraud_percentage=cfg.target_fraud_percentage, fraud_agents_count=cfg.fraud_agents_count, legit_agents_count=cfg.legit_agents_count, DATA_FILE=DATA_FILE)
    time_taken = round((time.time()-start_time)/60, 2)
    print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")
    visualize_json_success_rate()
    #visualize_rewards()