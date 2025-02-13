import ollama
import time
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import random
import re
import uuid
from IPython import embed
import pandas as pd
import matplotlib.pyplot as plt

#"Mixtral 8x7B": "mixtral"
models = {
   # "Llama3": "llama3",
    "Mistral": "mistral",
    "DeepSeek-R1": "deepseek-r1",
   # "Mixtral 8x7B": "mixtral",
    #"Llama3": "llama3"
}

# Pick a fraudulent strategy
user_id = "AI-12345"
global_clock='2025-03-01 09:00:00'
fraud_label = 1

import re
import json

def extract_json(text):
    """Extracts all JSON arrays from the LLM response, handling both single and multiple blocks.
       Returns extracted JSON and the number of retries.
    """
    retry_count = 0

    while retry_count < 3:  # Limit retries to avoid infinite loops
        matches = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', text, re.DOTALL)

        if not matches:
            matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)

        if not matches:
            retry_count += 1
            return 'retry', retry_count  # Return 'retry' and count of retries

        combined_activities = []

        for json_text in matches:
            json_text_cleaned = re.sub(r'//.*', '', json_text)  # Remove inline comments
            try:
                data = json.loads(json_text_cleaned)
                combined_activities.extend(data if isinstance(data, list) else [data])
                return combined_activities, retry_count  # Return JSON + retry count
            except json.JSONDecodeError:
                retry_count += 1

    return 'retry', retry_count  # If still failing after max retries

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
    )

def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
behaviors = list(fraudulent_strategies.keys())
fraud_label=1
# Store results
benchmark_results = []

n_trials=10
for model_name, model_id in models.items():
    for behavior in behaviors:
        strategy = fraudulent_strategies[behavior]
        prompt = build_generation_prompt(strategy, fraud_label, behavior, global_clock, user_id)
        for trial in range(n_trials):
            print(f"Testing {model_name}... (Trial {trial + 1}/{n_trials})")
            print(f"Testing {model_name}...")

            start_time = time.time()
            response = ollama.chat(
                model=model_id, 
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()

            response_time = round(end_time - start_time, 2)
            response_content = response['message']['content']

            # Extract JSON and retry count
            extracted_json, retry_count = extract_json(response_content)

            # Store results
            benchmark_results.append({
                "Model": model_name,
                "Profile type": behavior,
                "Trial": trial + 1,
                "Response Time (s)": response_time,
                "Valid JSON": extracted_json != "retry",
                "Retries": retry_count,  
                "Raw Response": response_content,
                "Generated Transactions": extracted_json if extracted_json != "retry" else "Invalid JSON output"
            })


# Convert to DataFrame and display
df_results = pd.DataFrame(benchmark_results)
# Save df_results
df_results.to_csv('benchmark_results.csv', index=False)
print(df_results)
# Compute statistics
stats_df = df_results.groupby(["Model", "Profile Type"]).agg({
    "Valid JSON": ["count", "sum", lambda x: 100 * (1 - x.mean())],  # Total, valid count, failure rate (%)
    "Retries": ["mean", "max"],  # Average & max retries
    "Response Time (s)": ["mean", "min", "max"]  # Response time stats
}).reset_index()

import pandas as pd
import matplotlib.pyplot as plt

# Read the benchmark results from CSV
df_results = pd.read_csv('../benchmark_results.csv')

# Compute statistics
stats_df = df_results.groupby("Model").agg({
    "Valid JSON": ["count", "sum", lambda x: 100 * (1 - x.mean())],  # Total, valid count, failure rate (%)
    "Retries": ["mean", "max"],  # Average & max retries
    "Response Time (s)": ["mean", "min", "max"]  # Response time stats
}).reset_index()

# Rename columns for clarity
stats_df.columns = [
    "Model", "Profile Type", "Total Trials", "Valid JSON Count", "Failure Rate (%)",
    "Avg Retries", "Max Retries", "Avg Response Time (s)", "Min Response Time (s)", "Max Response Time (s)"
]

print(stats_df)
# Create a single figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot 1: Failure Rate Comparison
axes[0].bar(stats_df["Model"], stats_df["Failure Rate (%)"], alpha=0.75)
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Failure Rate (%)")
axes[0].set_title("Failure Rate Comparison Across Models")
axes[0].set_ylim(0, 100)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: Response Time Distribution
axes[1].bar(stats_df["Model"], stats_df["Avg Response Time (s)"], label="Avg", color="orange", alpha=0.65)
axes[1].scatter(stats_df["Model"], stats_df["Min Response Time (s)"], color="green", label="Min", marker="o")
axes[1].scatter(stats_df["Model"], stats_df["Max Response Time (s)"], color="red", label="Max", marker="o")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Response Time (s)")
axes[1].set_title("Response Time Distribution Across Models")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 3: Average Retries per Model
axes[2].bar(stats_df["Model"], stats_df["Avg Retries"], color="orange", alpha=0.65)
axes[2].set_xlabel("Model")
axes[2].set_ylabel("Average Retries")
axes[2].set_title("Average Number of Retries per Model")
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("benchmark_analysis.png")

# Display the figure
plt.show()


