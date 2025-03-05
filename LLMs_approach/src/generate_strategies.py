import json
import os
import random
import ollama

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

def generate_fraud_strategy(fraud_type=None, filename="strategies/fraud_strategies.json"):
    """
    Generates a concise JSON strategy for a specified fraud type
    (e.g., "Account Takeover," "Insider Trading") that your second LLM (Mistral)
    can use to generate fraudulent transactions.
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
You are an expert in simulating fraudulent banking behaviors.

Provide a concise JSON object describing a fraud strategy for: "{fraud_type}".

### Output Requirements:
- Output MUST be valid JSON only (no extra commentary).
- Fields to include in your JSON:
  1. "profile_or_fraud_type": string  # e.g. "Insider Trading","Account Takeover"
  2. "transaction_types_involved": array of strings  # e.g. ["Unauthorized Stock Sale","Transfer Out","Withdrawal"]
  3. "typical_amount_range": string  # e.g. "$5,000 - $50,000"
  4. "geographic_focus": string or array  # e.g. ["Hong Kong","London","New York"]
  5. "velocity": string  # describes frequency, e.g. "multiple transactions within 24 hours"
  6. "currency": string or array  # e.g. "USD","CNY"
  7. "common_devices": array of strings  # typical compromised devices used
  8. "ip_address_notes": string  # e.g. "Often proxies from 203.x.x.x or 45.x.x.x"
  9. "context": string  # short narrative about how the fraud typically unfolds

### Example Format (do NOT copy verbatim):
{{
  "profile_or_fraud_type": "Account Takeover",
  "transaction_types_involved": ["Withdrawal","Transfer Out","Purchase"],
  "typical_amount_range": "$50,000 - $500,000",
  "geographic_focus": ["Domestic US, occasional international in China or Singapore"],
  "velocity": "High velocity: multiple transactions in under an hour",
  "currency": ["USD","CNY","SGD"],
  "common_devices": ["Windows 7 PC","iPhone-13 (stolen)"],
  "ip_address_notes": "Often uses compromised IP addresses from 45.x.x.x range",
  "context": "Fraudster gains access to victim accounts, executes quick, large transactions..."
}}

Return ONLY valid JSON with these nine fields, and no additional text.
"""

    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
    )
    strategy_text = response['message']['content'].strip()

    # Optionally validate the JSON here:
    # try:
    #     json.loads(strategy_text)
    # except json.JSONDecodeError:
    #     # handle or log error, or attempt a retry

    # Save to your file
    save_strategy_to_json(fraud_type, strategy_text, filename)
    return strategy_text

def generate_legitimate_strategy(profile_type=None, filename="strategies/legitimate_strategies.json"):
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
  1. "profile_or_fraud_type": string  # e.g. "Traveler", "Saver", "Investor"
  2. "transaction_types_involved": array of strings  # e.g. ["Purchase","Withdrawal","Transfer Out"]
  3. "typical_amount_range": string  # e.g. "$10 - $500"
  4. "geographic_focus": string or array  # e.g. "Domestic US" or ["New York, USA", "Shanghai, China"]
  5. "velocity": string  # describes frequency, e.g. "1-2 transactions per day"
  6. "currency": string or array  # e.g. "USD" or ["USD","EUR"]
  7. "common_devices": array of strings  # typical devices used
  8. "ip_address_notes": string  # typical IP range usage
  9. "context": string  # short narrative about how this profile usually behaves

### Example Format (not to be copied verbatim):
{{
  "profile_or_fraud_type": "Traveler",
  "transaction_types_involved": ["Purchase","Withdrawal"],
  "typical_amount_range": "$10 - $300",
  "geographic_focus": ["Asia","Europe"],
  "velocity": "About 2 transactions per day",
  "currency": ["USD","EUR","JPY"],
  "common_devices": ["iPhone-12","MacBook Air"],
  "ip_address_notes": "Mostly US-based IP (73.x.x.x), occasional foreign IP (203.x.x.x)",
  "context": "Frequently travels internationally, making small daily purchases and occasional larger withdrawals..."
}}

Return ONLY valid JSON with these nine fields, and no additional commentary.
"""

    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}],
    )
    strategy_text = response['message']['content'].strip()

    # Optionally, you can do a quick validation/parsing of the JSON here:
    # try:
    #     json.loads(strategy_text)
    # except json.JSONDecodeError:
    #     print("Strategy JSON is malformed; consider retry or post-processing repair.")
    #     # ... handle error ...

    # Save the strategy to a JSON file
    save_strategy_to_json(profile_type, strategy_text, filename)

    return strategy_text



os.makedirs('strategies', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Generate strategies
n_strategies=3
for i in range(n_strategies):
    generate_fraud_strategy()
    generate_legitimate_strategy()
