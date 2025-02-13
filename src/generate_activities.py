from datetime import datetime, timedelta, timezone
import time
import json
import os
import random
import re
import uuid
import pandas as pd
import ollama
import matplotlib.pyplot as plt

LLM_model = 'mistral'
ERROR_LOG_FILE = f"outputs/json_errors_{LLM_model}.log"
REWARD_LOG_FILE = f"outputs/reward_progress_{LLM_model}.csv"
log_text_file = f"outputs/llm_chain_of_thought_{LLM_model}.txt"
output_file = f"outputs/bank_log_{LLM_model}.csv"

# Initialize reward tracking
def initialize_reward_log():
    if not os.path.exists(REWARD_LOG_FILE):
        with open(REWARD_LOG_FILE, 'w') as file:
            file.write("timestamp,reward\n")

def update_reward_log(success: bool):
    reward = 1 if success else -1
    with open(REWARD_LOG_FILE, 'a') as file:
        file.write(f"{datetime.now().isoformat()},{reward}\n")

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
    update_reward_log(success=False)

def extract_json(text, user_id):
    """Extracts JSON arrays from LLM responses and validates/corrects them before parsing."""
    matches = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', text, re.DOTALL)    
    if not matches:
        log_json_error("No valid JSON arrays found in the response.", text)
        update_reward_log(success=False)
        return 'retry'

    # ✅ Step 1: Validate the raw extracted JSON
    if not validate_json(matches):
        print(f"❌ Validation failed for user {user_id}. Attempting correction...") 
        # 🔄 Step 2: Attempt Correction
        corrected_json = correct_json(matches)
        if corrected_json and validate_json(corrected_json):
            print(f"✅ JSON successfully corrected for user {user_id}")
            matches = corrected_json
        else:
            print(f"❌ Correction failed for user {user_id}. Returning 'retry' for regeneration...")
            update_reward_log(success=False)
            return 'retry'
    combined_activities = []
    for match in matches:
        # ✅ Step 3: Parse JSON after validation/correction
        try:
            data = json.loads(match)
            combined_activities.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError as e:
            log_json_error(f"JSON Decode Error: {e}", match)
            update_reward_log(success=False)
            return 'retry'

    update_reward_log(success=True)
    return combined_activities if combined_activities else 'retry'




def validate_json(activity_sequence,):
    """Validates JSON correctness using an LLM and ensures no comments exist."""
    json_string = json.dumps(activity_sequence, indent=2)
    
    validation_prompt = f"""
    You are a JSON validator. Review the JSON for correctness:
    
    ```json
    {json_string}
    ```
    
    Validation Rules:
    - Ensure JSON is NOT empty.
    - JSON must **NOT contain comments** (`//` or `/* ... */`).
    - JSON syntax must be valid.
    - Verify that all required fields exist.
    - Ensure `balance_after = balance_before - amount` (unless transaction is denied).
    - Ensure timestamps are sequential.
    
    If the JSON is correct, return `"VALID"`. Otherwise, return a detailed error report.
    """

    response = ollama.chat(model="gemma-2b", messages=[{"role": "user", "content": validation_prompt}])
    validation_result = response['message']['content'].strip()

    if "VALID" in validation_result:
        update_reward_log(success=True)
        return True
    else:
        update_reward_log(success=False)
        log_json_error(validation_result, json_string)
        return False

def correct_json(activity_sequence):
    """Uses a small LLM to correct minor JSON errors."""
    correction_prompt = f"""
    You are an expert JSON corrector. Here is a JSON that contains errors:
    
    ```json
    {json.dumps(activity_sequence, indent=2)}
    ```
    
    Your task:
    - Fix any structural issues.
    - Ensure numerical consistency.
    - Return ONLY the corrected JSON, nothing else.
    """

    response = ollama.chat(model="phi-2", messages=[{"role": "user", "content": correction_prompt}])
    return response['message']['content'].strip()



def build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id):
    """Builds the prompt for generating activity sequences with dynamic values and data schema."""
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
    
    return f"""
    You are an AI generating **detailed sequences of banking activities** for a fraud simulation.
    
    ### Strategy:
    {strategy}

    If `{fraud_label}`=1:
      1. Ensure the sequence of activities **completes the fraud** as described.
      2. If the fraud is **multi-event**, generate at least **minimum_activities** transactions.
      3. If the fraud is **single-event**, ensure the transaction fully represents the fraudulent behavior.

    ### Data Generation Rules:
    The financial activities must be generated as a **JSON object** with the exact format below:

    {json_template}

    - **Ensure all field values are valid** (timestamps, locations, amounts, etc.).
    - **Output only JSON**, enclosed within **triple backticks** (```json ... ```).
    - **Do not include comments or explanations in the JSON**.
    - Ensure logical consistency (e.g., `balance_after = balance_before - amount`).
    """

def generate_activity_sequence(strategy, fraud_label=0, profile_type="Legitimate", global_clock=None, user_id=None):
    """Generates structured financial activities based on the provided strategy."""
    while True:
        prompt = build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id)
        response = ollama.chat(model=LLM_model, messages=[{"role": "user", "content": prompt}])
        raw_response = response['message']['content'].strip()
        save_to_text(raw_response, user_id)

        activity_sequence = extract_json(raw_response)

        if activity_sequence != 'retry':
            print('✅ Activity sequence generated successfully for user ID:', user_id)
            return activity_sequence 

        print("🔄 Retrying activity sequence generation due to invalid JSON...")

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
            expected_columns = [
                "transaction_id", "bank_timestamp", "local_timestamp", "user_id", "account_id",
                "type", "amount", "currency", "balance_before", "balance_after",
                "location", "ip_address", "device_id", "network_type", "merchant_name",
                "recipient_id", "recipient_bank", "granted", "login_attempts", "session_id",
                "velocity", "distance_from_last_location", "is_repeat_location",
                "fraud_label", "behavior_type"
            ]
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
initialize_reward_log()
generate_activities(total_activities=20, target_fraud_percentage=0.5, fraud_agents_count=2, legit_agents_count=2)
# Time required to generate activities in minutes approximated
time_taken = round(time.time()-start_time/60,2)
print(f"Dataset generation required time: {round((time.time()-start_time)/60,1)} minutes")  
visualize_rewards()