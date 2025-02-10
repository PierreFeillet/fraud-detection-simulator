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
    
    # Generate new strategy
    fraud_description = (
        f"You are a clever fraudster specializing in {fraud_type}. "
        f"Your goal is to execute a fraud strategy that banks might detect but in a way that minimizes your risk. "
        f"Describe your method realistically, limited to what a banking system can observe. "
        f"Outline the step-by-step approach, key financial activities, and tactics to avoid detection."
    )

    prompt = (
        f"You're in a banking simulation where fraud checks can be immediate alerts for High-Risk Transactions "
        f"(like a flagged large foreign withdrawal) and Continuous Monitoring of Activity Patterns to catch subtler fraud over time.\n"
        f"Your goal is to execute a fraudulent scheme.\n"
        f"{fraud_description} "
        "Your strategy will be used by another LLM to generate a sequence of financial activities aligned with your plan. "
        "Provide:\n"
        "- The goal of your fraud scheme\n"
        "- The specific fraud type (e.g., money laundering, account takeover, synthetic identity fraud, card skimming)\n"
        "- The key financial tactics used (e.g., multiple small transactions, shell companies, offshore transfers)\n"
        "- A structured plan outlining the activities needed to reach your goal\n"
        "Provide a clear thought process, but do NOT return JSON."
    )

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
        "Provide a clear thought process, but do NOT return JSON."
    )

    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    strategy_text = response['message']['content'].strip()

    save_strategy_to_json(profile_type, strategy_text, filename)

    return strategy_text


os.makedirs('strategies', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Generate a fraud strategy
n_strategies=5
for i in range(n_strategies):
    fraud_strategy = generate_fraud_strategy()
    generate_legitimate_strategy()