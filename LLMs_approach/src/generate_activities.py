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
from utilities import generate_random_hash, format_timestamp, update_balance, assign_initial_balance
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
    "is_hijacked",
    # Behavior and fraud info
    "behavior_type",
    "fraud_label"
]

history_by_account = {}  # Dictionary to store transaction history per account


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


def assign_actvity_fields(activity, user_id, behavior_type, is_hijacked, fraud_label):
    """Assigns additional fields to the activity."""
    activity['user_id'] = user_id
    activity['activity_id'] = f"TXN-{generate_random_hash(10)}"
    activity['behavior_type'] = behavior_type
    activity['is_hijacked'] = is_hijacked
    activity['fraud_label'] = fraud_label
    return activity


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

def build_generation_prompt(strategy, global_clock, user_id, history, balance, is_hijacked,past_errors=None):
    """
    Builds a prompt for generating the next activity.
    
    - The LLM must create an activity **consistent with past transactions**.
    - The activity **is not always a transaction**; it could be a login, password change, etc.
    - Ensures **time-series consistency** with a strictly increasing timestamp.
    """

    # 1️⃣ **Summarize the user's history**
    history_summary = ""
    if history:
        history_summary = f"Previous activities for user {user_id}:\n" + json.dumps(history[-5:], indent=2) + "\n"  # Last 5 activities
        last_timestamp = history[-1].get("bank_timestamp", global_clock)

        time_instruction = (
            f"Use the last bank_timestamp {last_timestamp} as a reference for the next activity.\n"
            "The new bank_timestamp must be strictly later (chronologically) than the last one by at least 1 minute.\n"
            "If the new bank_timestamp is equal to or earlier than the last one, add sufficient delay.\n"
            "Generate a plausible gap in time based on the activity type and the user's typical activity pattern."
        )
    else:
        # If no history, we start from the global clock.
        last_timestamp = global_clock
        time_instruction = (
            f"Use the global clock {global_clock} as a reference for the activity.\n"
            "The generated bank_timestamp should be close to this time."
        )

    # 2️⃣ **Provide a JSON template example**
    json_template = """
<<<JSON>>>        
{
  "bank_timestamp": "2025-03-01T10:15:32+00:00",
  "local_timestamp": "2025-03-01T05:15:32-05:00",
  "account_id": "ACC-82736401",
  "type": "Login Attempt",
  "balance_before": 1280.45,
  "amount": 0.00,
  "location": "Chicago, USA",
  "ip_address": "73.56.201.89",
  "device_id": "iPhone-13",
  "network_type": "Wi-Fi",
  "merchant_name": null,
  "recipient_id": null,
  "recipient_bank": null
}
<<<END_JSON>>>
"""

    # 3️⃣ **Explain the fields**
    field_explanation = (
        "- **bank_timestamp**: string (ISO 8601), the UTC time of the activity.\n"
        "- **local_timestamp**: string (ISO 8601), the local time with correct offset.\n"
        "- **account_id**: string, formatted as \"ACC-XXXXXXXX\".\n"
        "- **type**: string, can be **a financial transaction or a non-transactional activity** (e.g., login, password change, fraud alert).\n"
        "- **balance_before**: float, the balance before the activity.\n"
        f"- **amount**: float, within the strategy-defined range, must be **≤ {balance}**.\n"
        "- **location**: string, city and country.\n"
        "- **ip_address**: string, a valid IPv4 address.\n"
        "- **device_id**: string, the device model.\n"
        "- **network_type**: string, e.g., \"Wi-Fi\", \"Cellular\".\n"
        "- **merchant_name**: string (null if not applicable).\n"
        "- **recipient_id**: string formatted as \"REC-XXXXXXXX\" (null if not applicable).\n"
        "- **recipient_bank**: string (null if not applicable).\n"
    )

    # 4️⃣ **Build the final prompt**
    prompt_parts = []

    # If prior activities exist, summarize them
    if history_summary:
        prompt_parts.append(history_summary)
        prompt_parts.append(f"- The new bank_timestamp must be strictly after {last_timestamp} (at least 1 minute later).\n")

    else:
        prompt_parts.append(f"- No prior history. Use the global clock {global_clock} as a reference.\n")

    # Add time instruction
    prompt_parts.append(time_instruction + "\n")

    # **Ensure LLM maintains user behavior consistency**
    prompt_parts.append(f"You are generating the next probable banking activity for user {user_id} with strategy: {strategy}.\n")
    prompt_parts.append("The generated activity **must be consistent with previous activities** and reflect the user's transaction behavior.\n")

    # ✅ **Specify that activities are NOT always transactions**
    prompt_parts.append("Activities are not always transactions! They can include:\n")
    prompt_parts.append("- **Login Attempt** (e.g., from a new device or suspicious location)\n")
    prompt_parts.append("- **Password Reset** (e.g., triggered by a fraudster after an account takeover)\n")
    prompt_parts.append("- **Device Change** (e.g., switching from a laptop to a mobile device)\n")
    prompt_parts.append("- **Bank Account Update** (e.g., adding a new recipient for money laundering)\n")
    prompt_parts.append("- **Financial Transactions** (Purchases, Transfers, Withdrawals, etc.)\n")

    # Ensure LLM follows the JSON format
    prompt_parts.append(
        "The output must be **valid JSON** with **exactly** the same fields as this example:\n"
        f"{json_template}\n"
    )

    prompt_parts.append("**Do not include any extra fields, comments, or explanations.**\n")
    prompt_parts.append("**Only generate one activity per request.**\n")

    # 5️⃣ **Data generation rules**
    prompt_parts.append("### Data Generation Rules:\n")
    prompt_parts.append(f"- **Required fields and formats:**\n{field_explanation}\n\n")
    prompt_parts.append("- **Timestamps must be logically consistent** (UTC vs. local time offset).\n")
    prompt_parts.append("- **Ensure transaction amount is within strategy-defined range** and **≤ the latest account balance**. If the activity is a non-transactional activity, the amount is 0.\n")
    prompt_parts.append("- **Use realistic time differences** between activities.\n")

    # ✅ **For fraudulent agents, ensure realistic fraud behavior**
    if is_hijacked:
        prompt_parts.append(
            "- This account was originally legitimate but has been hijacked.\n"
            "- Fraud must be **gradual** to avoid detection: **Start with login attempts, small test transactions, and account updates before large transfers.**\n"
            "- Do not generate high-value transactions too soon!\n"
    )
    # ✅ **For legitimate agents, ensure consistent, low-risk behavior**
    prompt_parts.append("- If the user is a **legitimate agent**, generate normal banking activities.\n")

    # ✅ **Realistic IP address and location mapping**
    prompt_parts.append("- Derive `ip_address` from the transaction location:\n")
    prompt_parts.append("  - USA → US-based IPv4 ranges (73.x.x.x, 24.x.x.x)\n")
    prompt_parts.append("  - Europe → European IPv4 ranges (81.x.x.x, 217.x.x.x)\n")
    prompt_parts.append("  - China → Chinese IPv4 ranges (202.x.x.x, 223.x.x.x)\n")

    # ✅ **Prevent JSON formatting errors**
    prompt_parts.append("- **Ensure the generated JSON follows a flat structure** with **NO comments or extra text**.\n")
    prompt_parts.append("- **It is very important that the data is realistic!**\n")

    # ✅ **Include past errors to avoid repetition**
    if past_errors:
        prompt_parts.append(f"- Do not repeat previous errors: {past_errors}.\n")
    
    # ✅ **Return the full prompt**
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

def generate_activity_sequence(strategy, global_clock, user_id, behavior_type, fraud_label, num_activities=5, user_accounts=None, is_hijacked=False):
    """
    Generates a sequence of activities for an agent.

    - Fraudulent agents may either own accounts or hijack existing legitimate ones.
    - If hijacked, fraudulent transactions must start small and escalate gradually.
    - The LLM determines the amount based on the strategy range.
    """
    if isinstance(strategy, str):
        try:
            match = re.search(r'{.*}', strategy, re.DOTALL)
            if match:
                json_str = match.group(0)
            strategy_json = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in strategy: {strategy}")

    n_accounts = strategy_json.get("n_accounts", 1)
    accounts = {f"ACC-{generate_random_hash()}": assign_initial_balance(strategy_json) for _ in range(n_accounts)}
    print("Initialized user accounts:", accounts)

    activities = []

    for i in range(num_activities):
        if i == 0:
            account_id = random.choice(list(user_accounts.keys())) if user_accounts else random.choice(list(accounts.keys()))
            current_balance = accounts[account_id]
    

        # ✅ Get the transaction history for this account
        history = history_by_account.get(account_id, [])

        # ✅ Get the last transaction timestamp (or use global_clock if no history exists)
        last_timestamp = history[-1]["bank_timestamp"] if history else global_clock
        

        retries = 1
        while True:
            print(f"User {user_id}, Account {account_id}: Generating activity {i}/{num_activities} attempt {retries} with balance {current_balance:.2f}...")

            past_errors = read_past_errors()
            prompt = build_generation_prompt(strategy, global_clock, user_id, history, current_balance, is_hijacked, past_errors)

            # ✅ LLM determines the transaction with a valid amount range
            raw_response = watsonx_chat(prompt=prompt, model_id=activity_model_id, parameters=watsonx_helper.parameters_activity)
            save_to_text(raw_response, user_id)

            activity, errors = validate_json(raw_response, user_id, last_timestamp)
            if activity != 'retry':
                tx = activity[0]
                new_account_id = tx.get("account_id", account_id)

                if new_account_id not in user_accounts:
                    default_balance = random.randint(100, 10000)
                    print(f"New fraudulent account detected: {new_account_id}. Creating account with initial balance {default_balance:.2f}.")
                    user_accounts[new_account_id] = default_balance
                    history_by_account[new_account_id] = []

                current_balance = round(user_accounts[new_account_id], 2)

                tx["balance_before"] = current_balance
                if fraud_label == 1 and is_hijacked:
                    last_fraud_amounts = [tx["amount"] for tx in history if tx["fraud_label"] == 1]
                    
                    # If LLM generates an amount too high, reduce it based on history
                    if last_fraud_amounts:
                        avg_fraud = sum(last_fraud_amounts) / len(last_fraud_amounts)
                        max_allowed = avg_fraud * 3  # Limit large jumps
                        if tx["amount"] > max_allowed:
                            print(f"⚠️ LLM generated {tx['amount']}, exceeding fraud pattern. Adjusting to max {max_allowed:.2f}.")
                            tx["amount"] = round(random.uniform(avg_fraud, max_allowed), 2)

                new_balance = round(update_balance(tx), 2)
                tx["balance_after"] = new_balance
                tx = assign_actvity_fields(tx, user_id, behavior_type, is_hijacked,fraud_label)

                # ✅ Store the transaction in history
                if new_account_id not in history_by_account:
                    history_by_account[new_account_id] = []
                history_by_account[new_account_id].append(tx)

                user_accounts[new_account_id] = new_balance
                tx['bank_timestamp'] = format_timestamp(tx['bank_timestamp'])
                tx['local_timestamp'] = format_timestamp(tx['local_timestamp'])
                activities.append(tx)

                print(f"Activity generated for account {new_account_id}. New balance: {new_balance:.2f}")
                update_reward_log(score=1, user_id=user_id, reason="JSON generation successful")
                embed()
                break
            else:
                print(f"Retrying activity generation for user {user_id}, account {account_id}...")
                update_reward_log(score=-1, user_id=user_id, reason="Invalid JSON")
                retries += 1

    return activities



FRAUD_INJECTION_PROBABILITY = 1  # 10% of legitimate accounts get hijacked.

def generate_activities(total_activities=1000, target_fraud_percentage=0.1, fraud_agents_count=5, legit_agents_count=20, DATA_FILE='output_data.csv'):
    """
    Generates banking transactions with mixed fraud and legitimate behavior.

    - Some fraudsters create their own accounts.
    - Some fraudsters hijack existing accounts (gradual fraud escalation).
    - The LLM determines transaction amounts based on fraud strategy.
    """
    fraudulent_strategies = load_existing_strategies(f"strategies/fraud_strategies_{strategy_model}.json")
    legitimate_strategies = load_existing_strategies(f"strategies/legitimate_strategies_{strategy_model}.json")

    global_clock = datetime.now(timezone.utc).isoformat()
    total_generated = 0
    target_fraud = int(total_activities * target_fraud_percentage)

    # --- Fraudulent Activities (Pure Fraudsters) ---
    while total_generated < target_fraud:
        remaining = target_fraud - total_generated
        for _ in range(fraud_agents_count):
            if total_generated >= target_fraud:
                break
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(fraudulent_strategies.keys()))
            strategy = fraudulent_strategies[behavior_type]
            num_act = random.randint(1, 6)
            num_act = min(num_act, remaining)
            activities = generate_activity_sequence(
                strategy=strategy, 
                global_clock=global_clock, 
                user_id=user_id, 
                behavior_type=behavior_type, 
                fraud_label=1, 
                num_activities=num_act,
                user_accounts={},  # Fraudsters start with fresh accounts
                is_hijacked=False  # Pure fraudster
            )
            for activity in activities:
                flush_buffer_immediate(activity, DATA_FILE)
                total_generated += 1
                if total_generated >= target_fraud:
                    break

    # --- Legitimate Activities with Fraud Injection ---
    while total_generated < total_activities:
        for _ in range(legit_agents_count):
            if total_generated >= total_activities:
                break
            user_id = f"USER-{generate_random_hash(8)}"
            behavior_type = random.choice(list(legitimate_strategies.keys()))
            strategy = legitimate_strategies[behavior_type]
            num_act = random.randint(1, 6)
            num_act = min(num_act, total_activities - total_generated)
            is_hijacked = random.random() < FRAUD_INJECTION_PROBABILITY
            fraud_label = 1 if is_hijacked else 0
            activities = generate_activity_sequence(
                strategy=strategy, 
                global_clock=global_clock, 
                user_id=user_id, 
                behavior_type=behavior_type, 
                fraud_label=fraud_label, 
                num_activities=num_act,
                user_accounts={},  # Use existing legitimate accounts
                is_hijacked=is_hijacked
            )
            for activity in activities:
                flush_buffer_immediate(activity, DATA_FILE)
                total_generated += 1
                if total_generated >= total_activities:
                    break

    print(f"Activity generation complete. Data saved to {DATA_FILE}")

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