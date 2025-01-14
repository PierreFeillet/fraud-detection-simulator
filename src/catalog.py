# Behavioral catalog for various fraudulent behaviors with time series and activity sequences
import numpy as np
import random

# Locations
locations = ["New York, USA", "Los Angeles, USA", "London, UK", "Tokyo, Japan", "Paris, France", "Berlin, Germany"]
# Define location weights (if you want some locations to appear more frequently)
location_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Adjust these weights as needed
LOCATIONS = {
    "USA": [
        ("New York", 0.2),
        ("Los Angeles", 0.2)
    ],
    "UK": [
        ("London", 0.15)
    ],
    "France": [
        ("Paris", 0.1)
    ],
    "Italy": [
        ("Rome", 0.1)
    ],
    "Germany":[
        ("Berlin", 0.1)
    ],
    "Japan":[
        ("Tokyo", 0.15)
    ]
}

# Merchants
# Define allowed transaction types for each merchant
TRANSACTION_TYPE_MERCHANTS = {
    "purchase": [
        ("Walmart", 0.15),
        ("Best Buy", 0.2),
        ("Target", 0.1),
        ("Starbucks", 0.1),
        ("Apple", 0.1),
        ("Amazon", 0.25),
        ("PayPal", 0.1)
    ],
    "deposit": [
        ("Bank", 0.4),
        ("ATM", 0.3),
        ("PayPal", 0.3)
    ],
    "withdrawal": [
        ("Bank", 0.5),
        ("ATM", 0.5)
    ]
}

# Devices and device weights
devices = ["iPhone", "Android", "Windows Laptop", "MacBook", "Linux PC", "iPad"]
device_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]  # Weights representing how frequently each device is used

# Networks and network weights
networks = ["Home WiFi", "Public WiFi", "Mobile Network", "Corporate Network"]
network_weights = [0.4, 0.3, 0.2, 0.1]  # Weights indicating the frequency of each network

# List of transactions. If not in this list a certain operation is considered an activity
possible_transactions = ["withdrawal", "deposit", "purchase"]

# Behavioral catalog for different fraud types

behavioral_catalog = {
    "normal": {
        "transaction_behavior": {
            "withdrawal_range": (50, 500),
            "deposit_range": (100, 1000),
            "purchase_range": (0, 700),
            "transaction_frequency": [0.3, 0.5, 0.2]  # 30% withdrawals, 50% deposits, 20% purchase
        },
        "activity_behavior": {
            "sequence": ["password_change", "email_change", "phone_change"],
            "time_probabilities": [0.3, 0.3, 0.4]  # Normal customer activities
        },
        "fraud": 0
            },
    "identity_theft": {
        "transaction_behavior": {
            "withdrawal_range": (1000, 5000),
            "deposit_range": (10, 100),
            "purchase_range": (0, 700),
            "transaction_frequency": [0.7, 0.1, 0.2]  # withdrawals, deposit, purchase
        },
        "activity_behavior": {
            "sequence": ["failed_login", "failed_login", "password_change", "suspicious_login", "withdrawal",],
            "time_probabilities": [0.2, 0.2, 0.3, 0.7, 0.8,]  # Probability of activity happening at different times
        },
        "fraud": 1
    },
    "money_laundering": {
        "transaction_behavior": {
            "withdrawal_range": (2000, 10000),
            "deposit_range": (500, 5000),
            "purchase_range": (0, 700),
            "transaction_frequency": [0.6, 0.05, 0.35]  

        },
        "activity_behavior": {
            "sequence": ["deposit", "deposit", "withdrawal", "phone_change", "email_change"],
            "time_probabilities": [0.5, 0.6, 0.8, 0.4, 0.3]
        },
        "fraud": 1
    },
    "phishing": {
        "transaction_behavior": {
            "withdrawal_range": (1000, 3000),
            "deposit_range": (50, 200),
            "purchase_range": (0, 700),
            "transaction_frequency": [0.6, 0.1, 0.3]  # 30% withdrawals, 50% deposits, 20% purchase

        },
        "activity_behavior": {
            "sequence": ["failed_login", "suspicious_login", "password_change", "withdrawal", "phone_change"],
            "time_probabilities": [0.7, 0.6, 0.5, 0.8, 0.4]
        },
        "fraud": 1
    },
    "card_skimming": {
        "transaction_behavior": {
            "withdrawal_range": (500, 2000),
            "deposit_range": (0, 50),
            "purchase_range": (0, 700),
            "transaction_frequency": [0.5, 0.1, 0.4]  # 30% withdrawals, 50% deposits, 20% purchase

        },
        "activity_behavior": {
            "sequence": ["withdrawal", "failed_login", "withdrawal", "phone_change", "email_change"],
            "time_probabilities": [0.6, 0.5, 0.8, 0.3, 0.2]
        },
        "fraud": 1
    }
}

# Generate transaction amount based on behavior
def generate_transaction_amount(transaction_type, behavior_type, precision=2):
    if transaction_type == "withdrawal":
        amount = max(0, np.random.uniform(*behavioral_catalog[behavior_type]["transaction_behavior"]["withdrawal_range"]))
    elif transaction_type == "deposit":
        amount = max(0, np.random.uniform(*behavioral_catalog[behavior_type]["transaction_behavior"]["deposit_range"]))
    elif transaction_type == "purchase":
        amount = max(0, np.random.uniform(*behavioral_catalog[behavior_type]["transaction_behavior"]["purchase_range"]))
    else:
        amount = 0  # Default if transaction_type is unrecognized
    return round(amount, precision)  # Round to the specified precision


# Determine if the transaction is a withdrawal or deposit based on behavior
def get_transaction_type(behavior_type,):
    new_type = np.random.choice(
        ["withdrawal", "deposit", "purchase"],
        p=behavioral_catalog[behavior_type]["transaction_behavior"]["transaction_frequency"]
    )
    return new_type

def get_activity_type(behavior_type):
    activity_sequence = behavioral_catalog[behavior_type]["activity_behavior"]["sequence"]
    time_probabilities = behavioral_catalog[behavior_type]["activity_behavior"]["time_probabilities"]
    return np.random.choice(activity_sequence, p=check_and_normalize_probabilities(time_probabilities))

def is_fraud(behavior_type):
    return behavioral_catalog[behavior_type]["fraud"]

def extract_merchant(transaction):
    """
    Extract a random merchant based on the transaction type and probabilities.
    
    Args:
        transaction (str): The transaction type (e.g., 'purchase', 'deposit', 'withdrawal').
        
    Returns:
        str: The selected merchant.
    """
    if transaction not in TRANSACTION_TYPE_MERCHANTS:
        raise ValueError(f"Transaction type '{transaction}' is not recognized.")
    merchants, probabilities = zip(*TRANSACTION_TYPE_MERCHANTS[transaction]) # Unzip the dictionary for the given transaction to get an array of merchants and an array of relative probabilities
    merchant = np.random.choice(merchants, p=probabilities)
    return merchant


def check_and_normalize_probabilities(time_probabilities):
    total = sum(time_probabilities)
    if not np.isclose(total, 1.0):
        # If the sum is not close to 1, normalize the array
        time_probabilities = [p / total for p in time_probabilities]
    return time_probabilities



