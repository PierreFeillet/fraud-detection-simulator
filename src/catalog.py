# Behavioral catalog for various fraudulent behaviors with time series and activity sequences
import numpy as np
import random

# Locations
locations = ["New York, USA", "Los Angeles, USA", "London, UK", "Tokyo, Japan", "Paris, France", "Berlin, Germany"]
# Define location weights (if you want some locations to appear more frequently)
location_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Adjust these weights as needed

# Merchants
merchants = ["Amazon", "Walmart", "Best Buy", "Target", "Starbucks", "Apple Store"]
merchant_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]

# Devices and device weights
devices = ["iPhone", "Android", "Windows Laptop", "MacBook", "Linux PC", "iPad"]
device_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]  # Weights representing how frequently each device is used

# Networks and network weights
networks = ["Home WiFi", "Public WiFi", "Mobile Network", "Corporate Network"]
network_weights = [0.4, 0.3, 0.2, 0.1]  # Weights indicating the frequency of each network

# Behavioral catalog for different fraud types

behavioral_catalog = {
    "normal": {
        "transaction_behavior": {
            "withdrawal_range": (50, 500),
            "deposit_range": (100, 1000),
            "transaction_frequency": 0.4  # 40% withdrawals, 60% deposits
        },
        "activity_behavior": {
            "sequence": ["password_change", "email_change", "phone_change"],
            "time_probabilities": [0.3, 0.3, 0.4]  # Normal customer activities
        }
    },
    "identity_theft": {
        "transaction_behavior": {
            "withdrawal_range": (1000, 5000),
            "deposit_range": (10, 100),
            "transaction_frequency": 0.7  # 70% of the time it's a withdrawal
        },
        "activity_behavior": {
            "sequence": ["failed_login", "failed_login", "password_change", "suspicious_login", "withdrawal"],
            "time_probabilities": [0.2, 0.2, 0.3, 0.7, 0.8]  # Probability of activity happening at different times
        }
    },
    "money_laundering": {
        "transaction_behavior": {
            "withdrawal_range": (2000, 10000),
            "deposit_range": (500, 5000),
            "transaction_frequency": 0.4  # Less frequent large withdrawals
        },
        "activity_behavior": {
            "sequence": ["deposit", "deposit", "withdrawal", "phone_change", "email_change"],
            "time_probabilities": [0.5, 0.6, 0.8, 0.4, 0.3]
        }
    },
    "phishing": {
        "transaction_behavior": {
            "withdrawal_range": (1000, 3000),
            "deposit_range": (50, 200),
            "transaction_frequency": 0.6
        },
        "activity_behavior": {
            "sequence": ["failed_login", "suspicious_login", "password_change", "withdrawal", "phone_change"],
            "time_probabilities": [0.7, 0.6, 0.5, 0.8, 0.4]
        }
    },
    "card_skimming": {
        "transaction_behavior": {
            "withdrawal_range": (500, 2000),
            "deposit_range": (0, 50),
            "transaction_frequency": 0.8  # Frequent small withdrawals
        },
        "activity_behavior": {
            "sequence": ["withdrawal", "failed_login", "withdrawal", "phone_change", "email_change"],
            "time_probabilities": [0.6, 0.5, 0.8, 0.3, 0.2]
        }
    }
}

# Generate transaction amount based on behavior
def generate_transaction_amount(transaction_type, behavior_type, precision=2):
    if transaction_type == "withdrawal":
        amount = max(0, np.random.uniform(*behavioral_catalog[behavior_type]["transaction_behavior"]["withdrawal_range"]))
    elif transaction_type == "deposit":
        amount = max(0, np.random.uniform(*behavioral_catalog[behavior_type]["transaction_behavior"]["deposit_range"]))
    else:
        amount = 0  # Default if transaction_type is unrecognized
    
    return round(amount, precision)  # Round to the specified precision

# Determine if the transaction is a withdrawal or deposit based on behavior
def transaction_type(behavior_type):
    return np.random.choice(
        ["deposit", "withdrawal"],
        p=[1 - behavioral_catalog[behavior_type]["transaction_behavior"]["transaction_frequency"],
           behavioral_catalog[behavior_type]["transaction_behavior"]["transaction_frequency"]]
    )

def check_and_normalize_probabilities(time_probabilities):
    total = sum(time_probabilities)
    if not np.isclose(total, 1.0):
        # If the sum is not close to 1, normalize the array
        time_probabilities = [p / total for p in time_probabilities]
    return time_probabilities


# Generate account activities based on behavior type and sequence
def account_activity(behavior_type):
    activity_sequence = behavioral_catalog[behavior_type]["activity_behavior"]["sequence"]
    time_probabilities = behavioral_catalog[behavior_type]["activity_behavior"]["time_probabilities"]
    return np.random.choice(activity_sequence, p=check_and_normalize_probabilities(time_probabilities))
