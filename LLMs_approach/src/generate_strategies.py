import json
import os
import random
import ollama
from watsonx_helper import watsonx_chat
import watsonx_helper
import geography

model_id=watsonx_helper.strategy_gen_model_id
model = model_id.split("/")[0]
print(f"Using model: {model_id}")

LOG_TEXT_FILE = os.path.join("strategies", f"llm_chain_of_thought_{model}.txt")

# Fraudulent and Legitimate categories
TOP_10_FRAUD_TYPES = [
    "Money Laundering", "Account Takeover", "Synthetic Identity Fraud", "Identity Theft",
    "Card Skimming", "Loan Fraud", "Check Fraud", "Wire Fraud",
    "Ponzi Scheme", "Cryptocurrency Fraud", "Insider Trading"
]

TOP_10_LEGITIMATE_PROFILES = [
    "Saver", "Investor", "Traveler", "Everyday Spender",
    "Business Owner", "Student", "Retiree", "Frequent Online Shopper",
    "Tech Professional", "Freelancer"
]

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

def generate_strategy(strategy_type=None, strategy_category="fraudulent", filename=None):
    """
    Generates a structured JSON strategy for either a **fraudulent** or **legitimate** profile.
    Ensures the **geographic focus** is chosen from the predefined REGION_TO_CITIES.
    """

    if strategy_category == "fraudulent":
        strategy_list = TOP_10_FRAUD_TYPES
        if filename is None:
            filename = f"strategies/fraud_strategies_{watsonx_helper.strategy_gen_model_id.split('/')[0]}.json"
    elif strategy_category == "legitimate":
        strategy_list = TOP_10_LEGITIMATE_PROFILES
        if filename is None:
            filename = f"strategies/legitimate_strategies_{watsonx_helper.strategy_gen_model_id.split('/')[0]}.json"
    else:
        raise ValueError("Invalid strategy category! Choose 'fraudulent' or 'legitimate'.")

    # Load existing strategies
    existing_strategies = load_existing_strategies(filename)

    # Choose a random strategy type if not provided
    if strategy_type is None:
        strategy_type = random.choice(strategy_list)

    # If the strategy already exists, return it directly
    if strategy_type in existing_strategies:
        print(f"Strategy for {strategy_type} already exists.")
        return existing_strategies[strategy_type]
    else:
        print(f"Generating {strategy_type}...")

    # Enforce strict selection of geographic focus
    geographic_focus_options = list(geography.REGION_TO_CITIES.keys())

    example_fraud = f"""
    <<<JSON>>>
    {{
    "profile": "Insider Trading",
    "transaction_types_involved": ["Unauthorized Stock Purchase", "Unauthorized Stock Sale"],
    "typical_amount_range": "$5,000 - $50,000",
    "geographic_focus": ["Domestic US", "Europe"],
    "velocity": "High velocity: 3-8 transactions per hour",
    "common_devices": ["iPhone-13", "MacBook Pro"],
    "ip_address_notes": "Mostly US-based IPs (73.x.x.x) with occasional Asia-based proxies (203.x.x.x)",
    "common_merchant_names": ["Goldman Sachs", "JP Morgan", "Morgan Stanley"],
    "common_recipient_ids": ["REC-12345678", "REC-87654321"],
    "common_recipient_banks": ["Bank of America", "Wells Fargo", "Bankf of China"],
    "context": "This strategy exploits non-public information to execute quick, high-value trades. Transactions occur rapidly, often within an hour, with a mix of domestic and occasional international activities. Purchases and sales are common, and when transfers occur, typical recipient details follow the provided patterns."
    }}
    <<<END_JSON>>>
    """

    example_legitimate = f"""
    <<<JSON>>>
        {{
        "profile": "Saver",
        "transaction_types_involved": ["Purchase", "Withdrawal", "Deposit", "Transfer IN"],
        "typical_amount_range": "$5 - $200",
        "geographic_focus": ["Domestic US"],
        "velocity": "1-2 transactions per day",
        "common_devices": ["iPhone-13", "MacBook Pro"],
        "ip_address_notes": "Stable US-based IPs (e.g., 73.x.x.x)",
        "common_merchant_names": ["Starbucks", "Amazon", "Walmart"],
        "common_recipient_ids": ["REC-65397495", "REC-37591258"],
        "common_recipient_banks": ["Bank of America", "Wells Fargo", "Chase Bank"],
        "context": "This Saver profile is characterized by cautious spending habits and consistent monthly savings. Typical transactions include small purchases and occasional withdrawals, with most activity occurring domestically. The customer maintains an emergency fund and uses reliable devices and stable IP ranges for all transactions."
        }}
        <<<END_JSON>>>
    """

    # Construct the LLM prompt
    prompt = f"""
    You are an expert in simulating **realistic banking strategies**. Your task is to define the startegy for the profile "{strategy_type}" in a structured JSON format enclosed within <<<START_JSON>>> and <<<END_JSON>>> tags. The strategy should include the following details:\n

    1. **profile_or_fraud_type** (string) - The fraud type or legitimate banking profile.
    2. **transaction_types_involved** (array of strings) - E.g., ["Purchase", "Transfer Out", "Withdrawal"].
    3. **typical_amount_range** (string) - Format: "$X - $Y" (Example: "$10 - $500").
    4. **geographic_focus** (array of strings) - Must be selected from {geographic_focus_options}.
    5. **velocity** (string) - Use format: "X-Y transactions per hour/day/week" (Example: "3-5 transactions per day").
    6. **network_types** (array of strings) - Example: ["Wi-Fi", "Cellular"].
    7. **common_devices** (array of strings) - Example: ["iPhone-13", "MacBook Pro"].
    8. **ip_ranges** (array of strings) - Example: ["73.x.x.x", "203.x.x.x"].
    9. **common_merchant_names** (array of strings) - If applicable, list merchants (Example: ["Amazon", "Walmart"]).
    10. **common_recipient_ids** (array of strings) - Example: ["REC-", "REC-"].
    11. **common_recipient_banks** (array of strings) - Example: ["Bank of America", "Wells Fargo"].
    12. **context** (string) - Describe common behavior in **one sentence**.

    ### **Example Output Format**, don't copy verbatim:
    {example_fraud if strategy_category == "fraudulent" else example_legitimate}

    Return **ONLY** valid JSON with all fields. **Do NOT add any explanations or extra text**.
    """

    # Run on watsonx
    strategy_text = watsonx_chat(prompt=prompt, model_id=watsonx_helper.strategy_gen_model_id, parameters=watsonx_helper.parameters_strategy)

    # Save the strategy
    save_strategy_to_json(strategy_type, strategy_text, filename)
    print(f"Generated {strategy_category} strategy for {strategy_type}")

    return strategy_text


os.makedirs('strategies', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

#with open(LOG_TEXT_FILE, 'w') as f:
#    f.write("")

# Generate strategies
n_strategies=3
for i in range(n_strategies):
    generate_strategy(strategy_category="fraudulent")
    generate_strategy(strategy_category="legitimate")
