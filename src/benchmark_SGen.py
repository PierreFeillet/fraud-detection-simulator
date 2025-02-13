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

#"Mixtral 8x7B": "mixtral"
models = {
    "Mistral": "mistral",
    "DeepSeek-R1": "deepseek-r1"
}

# Pick a fraudulent strategy
strategy = "<think>\nAlright, so I need to come up with a detailed strategy for simulating a Ponzi scheme. Hmm, let me think about what a Ponzi scheme entails.\n\nFirst off, from what I know, a Ponzi scheme is when an operator pays returns to early investors using the profits from new investments rather than legitimate earnings. It's pyramid-shaped because each layer relies on recruiting more people at the next level down.\n\nSo, for simulation purposes, how can we model this? Maybe start by identifying the key components: recruitment of new investors, distribution of returns, and the depletion of funds to sustain the scheme.\n\nIn terms of transaction types, I think it would involve multiple deposits from early investors, which are then used to pay out larger sums to later investors. That way, each subsequent layer gets a bigger payout than the previous one, creating the pyramid effect.\n\nNow, considering the minimum number of activities needed\u2014probably at least 5 transactions: some initial deposits, maybe a few withdrawals for payouts, and a couple more as the scheme depletes. But I'm not sure if that's enough; perhaps it should involve more steps to show how each layer feeds into the next.\n\nTransaction amounts might vary, but early investors could have smaller deposits, while later ones receive larger payouts. The geographic pattern could be concentrated within one region at first, then maybe expand slightly for recruitment purposes.\n\nVelocity-wise, since this is a simulated activity, it's probably high because participants act quickly to recruit and withdraw funds before authorities notice. Distance between transactions might be short periods between recruiting more investors or making payouts.\n\nI'm also thinking about the context. The scheme would likely start with an announcement by someone close to the organization, maybe an executive, who then recruits others. These early investors would deposit their money expecting returns based on rumors of success. Then, as the scheme progresses, each subsequent layer relies on the previous one's investments to sustain it.\n\nPotential red flags for fraud detection might include a concentration of deposits in certain locations followed by massive withdrawals. The scheme could also spread across multiple currencies or regions to make tracking difficult.\n\nBut wait, do I need to consider varying transaction amounts? Maybe early investors get smaller sums, while later ones receive exponentially larger amounts. That would help sustain the pyramid structure until the base is exhausted.\n\nAlso, considering that not all transactions might be legitimate\u2014some could be shell companies or front organizations facilitating the cash flows. This adds another layer of complexity for detection because it's harder to trace the actual sources and sinks of funds.\n\nI should also think about how the simulation will look in terms of activities. Each activity would involve a series of deposits, followed by withdrawals based on those deposits. Maybe some activities are used to show the growth of the pyramid, while others demonstrate the depletion phase as it becomes unsustainable.\n\nIn summary, to simulate a Ponzi scheme, I need to create multiple deposit transactions from early investors, then several withdrawal transactions using these funds for later investors. The number of required activities would depend on how many layers are in the pyramid and when the scheme becomes unmanageable. Including some shell companies or front organizations could make it more realistic.\n\nI think aiming for around 10 to 15 activities might be sufficient to show both the initial growth and the potential collapse as the scheme depletes its funds. The transactions should have varying amounts, starting smaller and increasing exponentially. Geographically, maybe concentrated in a few regions but with some international involvement to hide tracks.\n\nVelocity would likely be high since these are simulated activities meant to mimic real-time recruitment and fund movements. The distance between transactions might be short periods to allow quick growth before the scheme becomes too large for sustainability.\n\nI should also include red flags like unusual patterns in account creation, a sudden spike in withdrawals compared to deposits, and maybe some discrepancies in transaction sources or destinations that could indicate front organizations.\n\nOverall, this approach would help create a realistic simulation of a Ponzi scheme for fraud detection training.\n</think>\n\n```json\n{\n  \"Fraud Type\": \"Ponzi Scheme\",\n  \"Scope\": \"Multi-event\",\n  \"Minimum Activities\": 10,\n  \"Description\": {\n    \"Transaction Types Involved\": [\"Deposits from Early Investors\", \"Withdrawals to Later Investors\"],\n    \"Typical Transaction Amounts\": [\"Early investors: $5,000 - $50,000; Later investors: $25,000 - $100,000 per transaction\"],\n    \"Geographic Patterns\": [\"Concentrated within the organization's region with occasional international involvement for recruitment\"],\n    \"Velocity\": \"High velocity due to rapid execution of transactions to capitalize on timely information and recruitment\"],\n    \"Distance Between Transactions\": \"Minutes to hours between key activities like recruiting new investors or making payouts\"]\n  },\n  \"Context\": \"The Ponzi Scheme begins with an executive announcement promising high returns. Early investors deposit funds expecting significant returns. Subsequent layers rely on the previous investments for payouts, creating a pyramid structure. The scheme accelerates as each layer depends on the prior one's investments. Red flags include concentrated deposits followed by massive withdrawals and anomalies in account creation timing.\"\n}\n```"
profile_type = "Ponzi Scheme"
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

prompt = build_generation_prompt(strategy, fraud_label, profile_type, global_clock, user_id)
# Store results
benchmark_results = []

for model_name, model_id in models.items():
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
        "Response Time (s)": response_time,
        "Valid JSON": extracted_json != "retry",
        "Retries": retry_count,  # NEW: Track retry attempts
        "Raw Response": response_content,
        "Generated Transactions": extracted_json if extracted_json != "retry" else "Invalid JSON output"
    })

# Convert to DataFrame and display
df_results = pd.DataFrame(benchmark_results)
# Save df_results
df_results.to_csv('benchmark_results.csv', index=False)
print(df_results)
