import json
import os
import random
import ollama
from watsonx_helper import watsonx_chat
import watsonx_helper

model_id=watsonx_helper.strategy_gen_model_id
model = model_id.split("/")[0]
print(f"Using model: {model_id}")

LOG_TEXT_FILE = os.path.join("strategies", f"llm_chain_of_thought_{model}.txt")

#  Top 10 Banking Frauds
TOP_10_FRAUD_TYPES = [
    "Money Laundering", "Account Takeover", "Synthetic Identity Fraud", "Identity Theft",
    "Card Skimming", "Loan Fraud", "Check Fraud", 
    "Wire Fraud", "Ponzi Scheme", "Cryptocurrency Fraud", "Insider Trading"
]

#  Top 10 Legitimate Banking Profiles
TOP_10_LEGITIMATE_PROFILES = [
    "Saver", "Investor", "Traveler", "Everyday Spender", 
    "Business Owner", "Student", "Retiree", "Frequent Online Shopper",
    "Tech Professional", "Freelancer"
]
def save_to_text(reasoning_text, profile):
    """Appends LLM reasoning and extracted JSON to a shared text file."""
    with open(LOG_TEXT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### LLM Chain of Thought for strategy {profile} ###\n\n{reasoning_text}\n\n")

def load_existing_strategies(filename):
    """Loads existing strategies from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def save_strategy_to_json(strategy_type, strategy_text, filename):
    """Saves a strategy to a JSON file."""
    strategies = load_existing_strategies(filename)
    strategies[strategy_type] = strategy_text
    with open(filename, 'w') as file:
        json.dump(strategies, file, indent=4)

def generate_fraud_strategy(fraud_type=None, filename=f"strategies/fraud_strategies_{model}.json"):
    """
    Generates a concise JSON strategy for a specified fraud type
    (e.g., "Account Takeover," "Insider Trading") that your second LLM (Mistral)
    can use to generate fraudulent activities.
    """

    # Load existing strategies from your JSON file
    existing_strategies = load_existing_strategies(filename)

    # If no fraud type was specified, pick one randomly
    if fraud_type is None:
        fraud_type = random.choice(TOP_10_FRAUD_TYPES)

    # If the strategy already exists, return it directly
    if fraud_type in existing_strategies:
        print(f"Strategy for {fraud_type} already exists.")
        return existing_strategies[fraud_type]

    # Build a refined JSON-only prompt for the LLM
    prompt = f"""
You are an expert in banking transaction simulation and fraud detection. Your task is to define a detailed strategy that will guide the generation of realistic banking activities for the profile {fraud_type}.

Please provide a valid JSON object (without any additional commentary or chain-of-thought) with the following fields:

1. "fraud_type": string  
   // For example: {fraud_type}

2. "transaction_types_involved": array of strings  
   // For example: ["Purchase", "Transfer Out", "Withdrawal"]

3. "typical_amount_range": string  
   // For example: "$10 - $500" or "$5,000 - $50,000"

4. "geographic_focus": string or array  
   // For example: "Domestic US" or ["New York, USA", "Shanghai, China"]

5. "velocity": string  
   // For example: "1-2 transactions per day" or "Multiple transactions within 1 hour"

6. "currency": string or array  
   // For example: "USD" or ["USD", "CNY"]

7. "common_devices": array of strings  
   // For example: ["iPhone-13", "MacBook Pro"]

8. "ip_address_notes": string  
   // For example: "Mostly US-based IP ranges like 73.x.x.x, occasionally foreign IPs like 203.x.x.x"

9. "common_merchant_names": array of strings  
   // List common merchant names relevant for purchase or sale transactions (e.g., ["Starbucks", "Amazon", "Walmart"]).  
   // For transactions like Transfers, this field may be null.

10. "common_recipient_ids": array of strings  
    // Provide typical formats or examples for recipient IDs (e.g., ["REC-12345678", "REC-87654321"]).

11. "common_recipient_banks": array of strings  
    // Provide common recipient banks or patterns (e.g., ["Bank of America", "Wells Fargo", "BANK-XYZ"]).

12. "context": string  
    // A short narrative explaining how this strategy typically unfolds, highlighting key behaviors and any potential anomalies.

13. The JSON object must be enclosed withing the delimiter <<<JSON>>> ... <<<END_JSON>>> as in the example.

### Example Format (Do not copy verbatim; follow the structure):
<<<JSON>>>
{{
  "profile_or_fraud_type": "Insider Trading",
  "transaction_types_involved": ["Unauthorized Stock Purchase", "Unauthorized Stock Sale"],
  "typical_amount_range": "$5,000 - $50,000",
  "geographic_focus": ["Domestic US", "Occasional international trades"],
  "velocity": "High velocity: multiple transactions within 1 hour",
  "currency": ["USD", "HKD"],
  "common_devices": ["iPhone-13", "MacBook Pro"],
  "ip_address_notes": "Mostly US-based IPs (73.x.x.x) with occasional Asia-based proxies (203.x.x.x)",
  "common_merchant_names": ["Goldman Sachs", "JP Morgan", "Morgan Stanley"],
  "common_recipient_ids": ["REC-12345678", "REC-87654321"],
  "common_recipient_banks": ["Bank of America", "Wells Fargo", "BANK-XYZ"],
  "context": "This strategy exploits non-public information to execute quick, high-value trades. Transactions occur rapidly, often within an hour, with a mix of domestic and occasional international activities. Purchases and sales are common, and when transfers occur, typical recipient details follow the provided patterns."
}}
<<<END_JSON>>>
Return ONLY valid JSON with these exact fields.
""" 
    # To run lcoally with ollama
    #response = ollama.chat(
    #    model="deepseek-r1",
    #    messages=[{"role": "user", "content": prompt}],
    #)
    #strategy_text = response['message']['content'].strip()

    # Run on watsonx
    strategy_text = watsonx_chat(prompt=prompt, model_id=watsonx_helper.strategy_gen_model_id, parameters=watsonx_helper.parameters_strategy)
    # Save to your file
    save_strategy_to_json(fraud_type, strategy_text, filename)
    print(f"Generated strategy for {fraud_type}")
    save_to_text(strategy_text, fraud_type)
    return strategy_text

def generate_legitimate_strategy(profile_type=None, filename=f"strategies/legitimate_strategies_{model}.json"):
    """
    Generates a legitimate banking strategy based on a user profile (e.g. 'Saver', 'Traveler', etc.)
    and saves it in JSON format that can be used by another LLM (Mistral).
    """

    # Load existing strategies
    existing_strategies = load_existing_strategies(filename)

    # Pick a profile type if none is provided
    if profile_type is None:
        profile_type = random.choice(TOP_10_LEGITIMATE_PROFILES)

    # If the strategy already exists, return it directly
    if profile_type in existing_strategies:
        print(f"Strategy for {profile_type} already exists.")
        return existing_strategies[profile_type]

    # Build a refined prompt that requests JSON-only output with the needed fields
    prompt = f"""
You are an expert in simulating realistic, legitimate banking customer behaviors.

Provide a concise JSON object describing a legitimate customer profile of type: "{profile_type}".

### Output Requirements:
- Output MUST be valid JSON only (no extra commentary, no chain-of-thought).
- Fields to include in your JSON:
1. "profile": string  
   // For example: {profile_type}

2. "transaction_types_involved": array of strings  
   // For example: ["Purchase", "Transfer Out", "Withdrawal"]

        3. "typical_amount_range": string  
        // For example: "$10 - $500" or "$5,000 - $50,000"

        4. "geographic_focus": string or array  
        // For example: "Domestic US" or ["New York, USA", "Shanghai, China"]

        5. "velocity": string  
        // For example: "1-2 transactions per day" or "Multiple transactions within 1 hour"

        6. "currency": string or array  
        // For example: "USD" or ["USD", "CNY"]

        7. "common_devices": array of strings  
        // For example: ["iPhone-13", "MacBook Pro"]

        8. "ip_address_notes": string  
        // For example: "Mostly US-based IP ranges like 73.x.x.x, occasionally foreign IPs like 203.x.x.x"

        9. "common_merchant_names": array of strings  
        // List common merchant names relevant for purchase or sale transactions (e.g., ["Starbucks", "Amazon", "Walmart"]).  
        // For transactions like Transfers, this field may be null.

        10. "common_recipient_ids": array of strings  
            // Provide typical formats or examples for recipient IDs (e.g., ["REC-12345678", "REC-87654321"]).

        11. "common_recipient_banks": array of strings  
            // Provide common recipient banks or patterns (e.g., ["Bank of America", "Wells Fargo", "BANK-XYZ"]).

        12. "context": string  
            // A short narrative explaining how this strategy typically unfolds, highlighting key behaviors and any potential anomalies.
        
        13. The JSON object must be enclosed withing the delimiter <<<JSON>>> ... <<<END_JSON>>> as in the example.

        ### Example Format (not to be copied verbatim):
        <<<JSON>>>
        {{
        "profile": "Saver",
        "transaction_types_involved": ["Purchase", "Withdrawal", "Deposit", "Transfer IN"],
        "typical_amount_range": "$5 - $200",
        "geographic_focus": ["Domestic US"],
        "velocity": "1-2 transactions per day",
        "currency": "USD",
        "common_devices": ["iPhone-13", "MacBook Pro"],
        "ip_address_notes": "Stable US-based IPs (e.g., 73.x.x.x)",
        "common_merchant_names": ["Starbucks", "Amazon", "Walmart"],
        "common_recipient_ids": [],
        "common_recipient_banks": [],
        "context": "This Saver profile is characterized by cautious spending habits and consistent monthly savings. Typical transactions include small purchases and occasional withdrawals, with most activity occurring domestically. The customer maintains an emergency fund and uses reliable devices and stable IP ranges for all transactions."
        }}
        <<<END_JSON>>>

        Return ONLY valid JSON with these nine fields, and no additional commentary.
"""

        # To run lcoally with ollama
    #response = ollama.chat(
    #    model="deepseek-r1",
    #    messages=[{"role": "user", "content": prompt}],
    #)
    #strategy_text = response['message']['content'].strip()

    # Run on watsonx
    strategy_text = watsonx_chat(prompt=prompt, model_id=model_id, parameters=watsonx_helper.parameters_strategy)

    # Save to your file
    save_strategy_to_json(profile_type, strategy_text, filename)
    print(f"Generated strategy for {profile_type}")

    return strategy_text



os.makedirs('strategies', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

#with open(LOG_TEXT_FILE, 'w') as f:
#    f.write("")

# Generate strategies
n_strategies=3
for i in range(n_strategies):
    generate_fraud_strategy()
    generate_legitimate_strategy()
