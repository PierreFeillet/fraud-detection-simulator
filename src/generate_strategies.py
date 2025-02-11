import json
import os
import random
import ollama

#  Top 10 Banking Frauds
TOP_10_FRAUD_TYPES = [
    "Money Laundering", "Account Takeover", "Synthetic Identity Fraud",
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
    """Generates a fraud strategy based on a specific fraud type (or picks one randomly)."""    
    # Load existing strategies
    existing_strategies = load_existing_strategies(filename)
    
    # Pick a fraud type if none is provided
    if fraud_type is None:
        fraud_type = random.choice(TOP_10_FRAUD_TYPES)
    
    # Check if strategy already exists
    if fraud_type in existing_strategies:
        print(f"Strategy for {fraud_type} already exists.")
        return existing_strategies[fraud_type]
    
    def build_strategy_prompt(fraud_type):
        """Prompt for the Strategy LLM to define fraud patterns with structured JSON and descriptive context."""

        return f"""
        You are an expert in banking fraud detection tasked with defining detailed strategies for simulating fraudulent activities.

        ### Fraud Type:
        - **{fraud_type}**

        ### Instructions:
        1. Clearly describe the fraudulent behavior.
        2. Specify whether the fraud is:
        - **Single-event**: A one-time transaction that completes the fraud.
        - **Multi-event**: Requires multiple transactions to fully execute and reveal the fraud pattern.
        3. Define the **minimum number of activities** required for the fraud to be recognizable. This number must be realistic and specific to the {fraud_type} you're simulating.
        4. Describe the key characteristics of the fraud pattern, including:
        - **Transaction Types Involved**: Specify whether transactions are purchases, withdrawals, transfers, or trades.
        - **Typical Transaction Amounts**: Provide realistic ranges for transaction amounts.
        - **Geographic Patterns**: Indicate if transactions occur locally, internationally, or in high-risk locations.
        - **Velocity**: Describe how quickly transactions occur (e.g., rapid succession or spaced over time).
        - **Distance Between Transactions**: Specify the time intervals between transactions (e.g., minutes, hours, days).

        ### Important:
        - Include a **'Context' section** in the JSON output where you provide a detailed description of how the fraud typically unfolds.
        - The **structured fields** will guide constraints, while the **context** provides deeper narrative for the activity sequence.
        - Output the strategy in **structured JSON format** as shown below. Do NOT include any additional text or explanation after the JSON.

        ### Example Output Format:
        ```json
        {{
        "Fraud Type": "{fraud_type}",
        "Scope": "Multi-event",
        "Minimum Activities": 5,
        "Description": {{
            "Transaction Types Involved": ["Unauthorized Stock Purchases", "Unauthorized Stock Sales"],
            "Typical Transaction Amounts": ["Between $5,000 and $50,000 per transaction"],
            "Geographic Patterns": ["Concentrated within the organization's region, occasional international trades"],
            "Velocity": "High velocity due to the need for quick execution based on non-public information",
            "Distance Between Transactions": "Minutes to hours between trades to capitalize on timely information"
        }},
        "Context": "Insider Trading involves employees or individuals with access to non-public information executing unauthorized stock trades. These trades often occur in rapid succession, with the individual purchasing stock before a major positive announcement or selling it before a negative one. The activity is characterized by sudden, unexplainable trading behavior inconsistent with the individual's usual patterns, often concentrated in the company's geographic region but may also include international trades to obscure detection."
        }}
        ```
        """

    prompt=build_strategy_prompt(fraud_type)
    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()
    
    save_strategy_to_json(fraud_type, strategy_text, filename)
    
    return strategy_text

def generate_legitimate_strategy(profile_type=None, filename="strategies/legitimate_strategies.json"):
    """Generates a legitimate banking strategy based on a customer profile type."""

    # Load existing strategies
    existing_strategies = load_existing_strategies(filename)

    # Pick a profile type if none is provided
    if profile_type is None:
        profile_type = random.choice(TOP_10_LEGITIMATE_PROFILES)

    # Check if strategy already exists
    if profile_type in existing_strategies:
        print(f"Strategy for {profile_type} already exists.")
        return existing_strategies[profile_type]

    # Generate new strategy
    profile_description = (
        f"You are a legitimate bank customer with a {profile_type} profile. "
        f"Describe your typical financial behavior, transactions, and approach to money management."
    )

    prompt = (
        f"You're in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions "
        f"(like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time. "
        f"Sometimes legitimate customer operations are not granted because of the bank alerts. "
        f"You are a legitimate customer. "
        f"{profile_description} "
        "Your strategy will be used by another LLM to generate a sequence of financial activities aligned with your plan. "
        "Provide:\n"
        "- The financial goal of your behavior\n"
        "- Your customer profile (e.g., Saver, Investor, Traveler, Everyday Spender)\n"
        "- The key financial habits (e.g., monthly savings, frequent small purchases, international spending)\n"
        "- A structured plan outlining the expected activities\n"
        "Provide a clear thought process."
    )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()

    save_strategy_to_json(profile_type, strategy_text, filename)

    return strategy_text


os.makedirs('strategies', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Generate strategies
n_strategies=2
for i in range(n_strategies):
    fraud_strategy = generate_fraud_strategy()
    generate_legitimate_strategy()