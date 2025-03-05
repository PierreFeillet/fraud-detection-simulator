import numpy as np
import json
import random

behavior_catalog = {
    "legitimate": {
        "activities": {
            "Open Account": "neutral",                  # No transaction at the start
            "Deposit Funds": "positive",             # Can vary from 0 to 10,000
            "Make Purchase": "negative",            # Purchases are negative (money spent)
            "Pay Bills": "negative",                 # Payments are negative (money spent)
            "Transfer Funds": "negative",           # Transfers are negative (money sent out)
            "Apply for Credit Card": "neutral",         # No transaction, just an application
            "Apply for Loan": "neutral",                # No transaction, just an application
            "Invest Money": "negative",          # Investments are negative (money spent)
            "Review activityments": "neutral",          # No transaction, review only
            "Close Account": "neutral"                  # No transaction at account closure
        },
        "transition_matrix": [
            [0.0, 0.5, 0.2, 0.1, 0.1, 0.02, 0.02, 0.02, 0.03, 0.01],
            [0.0, 0.4, 0.3, 0.1, 0.1, 0.03, 0.02, 0.02, 0.07, 0.01],
            [0.0, 0.1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0],
            [0.0, 0.1, 0.2, 0.5, 0.1, 0.03, 0.02, 0.02, 0.02, 0.0],
            [0.0, 0.2, 0.3, 0.1, 0.3, 0.02, 0.02, 0.02, 0.02, 0.01],
            [0.0, 0.1, 0.2, 0.2, 0.1, 0.3, 0.05, 0.02, 0.01, 0.01],
            [0.0, 0.1, 0.1, 0.1, 0.1, 0.02, 0.4, 0.1, 0.05, 0.02],
            [0.0, 0.05, 0.05, 0.1, 0.1, 0.02, 0.03, 0.5, 0.1, 0.05],
            [0.0, 0.05, 0.05, 0.15, 0.14, 0.02, 0.03, 0.05, 0.5, 0.01],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "time_limit": 60*180,  # Maximum 3 hours between events
        "fraud": 0,
    },
    "identity_theft": {
        "activities": {
            "Failed Login": "neutral",                  # No transaction
            "Suspicious Login": "neutral",         # To be suspicious it should be accompanied by a weird location     # No transaction
            "Change Password": "neutral",               # No transaction
            "Change Email": "neutral",                  # No transaction
            "Change Phone": "neutral",                  # No transaction
            "Request New Card": "neutral",              # No transaction
            "Unauthorized Transfer": "negative",  # Unauthorized transfers are negative (money stolen)
            "Close Account": "neutral",                  # No transaction at account closure
        },
        "transition_matrix": [
            [0.6, 0.3, 0.05, 0.02, 0.02, 0.0, 0.01, 0.0],
            [0.1, 0.5, 0.3, 0.05, 0.03, 0.02, 0.05, 0.0],
            [0.0, 0.1, 0.5, 0.2, 0.1, 0.05, 0.05, 0.0],
            [0.0, 0.05, 0.1, 0.5, 0.2, 0.1, 0.05, 0.0],
            [0.0, 0.05, 0.1, 0.2, 0.5, 0.1, 0.04, 0.01],
            [0.0, 0.02, 0.05, 0.1, 0.1, 0.5, 0.2, 0.03],
            [0.0, 0.01, 0.02, 0.05, 0.1, 0.1, 0.6, 0.12],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "time_limit": 60*30,  # Shorter time between suspicious activity
        "fraud": 1,
    },

    "card_skimming": {
        "activities": {
            "ATM Withdrawal": "negative",           # Withdrawals are negative (money taken out)
            "Online Purchase": "negative",            # Purchases are negative (money spent)
            "POS Purchase": "negative",              # In-person purchases are negative (money spent)
            "Balance Check": "neutral",                  # No transaction for balance check
            "Request New PIN": "neutral",                # No transaction for PIN request
            "Report Lost Card": "neutral",               # No transaction for report
            "Close Account": "neutral"                   # No transaction at account closure
        },
        "transition_matrix": [
            [0.0, 0.3, 0.3, 0.2, 0.1, 0.05, 0.05],
            [0.0, 0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
            [0.0, 0.3, 0.4, 0.1, 0.1, 0.05, 0.04],
            [0.1, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05],
            [0.0, 0.1, 0.1, 0.1, 0.4, 0.2, 0.1],
            [0.0, 0.05, 0.05, 0.05, 0.2, 0.5, 0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "time_limit": 60*15,  # Rapid transactions typical in skimming
        "fraud": 1,
    },

    "money_laundering": {
        "activities": {
            "Open Account": "neutral",                  # No transaction at account opening
            "Deposit Funds": "positive",           # Deposits can vary, but negative when spent
            "Wire Transfer": "negative",         # Wire transfers are negative (money sent)
            "Make Purchas": "negative", # Luxury goods are negative (money spent)
            "Cash Withdrawal": "negative",         # Withdrawals are negative (money taken out)
            "Invest in Assets": "negative",      # Investments are negative (money spent)
            "Close Account": "neutral"                  # No transaction at account closure
        },
        "transition_matrix": [
            [0.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02],
            [0.0, 0.2, 0.4, 0.2, 0.1, 0.05, 0.03],
            [0.0, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
            [0.0, 0.1, 0.1, 0.4, 0.2, 0.1, 0.1],
            [0.0, 0.1, 0.1, 0.2, 0.4, 0.1, 0.1],
            [0.0, 0.1, 0.1, 0.2, 0.1, 0.4, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "time_limit": 60*120,  # Frequent but spread-out transactions
        "fraud": 1,
    },

    "synthetic_identity_fraud": {
        "activities": {
            "Create Fake Identity": "neutral",          # No transaction for creating fake identity
            "Open Account": "neutral",                  # No transaction at account opening
            "Apply for Credit Card": "neutral",         # No transaction for credit card application
            "Apply for Loan": "neutral",                # No transaction for loan application
            "Make Purchase": "negative",         # Purchases are negative (money spent)
            "Max Out Credit": "negative",        # Maxing out credit is negative (money used)
            "Withdraw Funds": "negative",         # Withdrawals are negative (money taken out)
            "Close Account": "neutral"                  # No transaction at account closure
        },
       "transition_matrix": [
        [0.4, 0.3, 0.1, 0.1, 0.05, 0.01, 0.01, 0.03],  # From Create Fake Identity
        [0.3, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01],  # From Open Account
        [0.2, 0.3, 0.3, 0.1, 0.05, 0.02, 0.02, 0.02],  # From Apply for Credit Card
        [0.1, 0.2, 0.2, 0.4, 0.1, 0.02, 0.02, 0.02],  # From Apply for Loan
        [0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1],     # From Make Purchase
        [0.05, 0.05, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1],   # From Max Out Credit
        [0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1],   # From Withdraw Funds
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]      # From Close Account (End of activity)
    ],
        "time_limit": 60*60,  # Faster spending to exploit identity
        "fraud": 1,
    }
}

# Example fraud rates by country (percent of fraudulent transactions)
fraud_rates_by_country = {
    "Nigeria": 0.25,      # 25% probability of an activity being fraudulent
    "Russia": 0.20,       # 20%
    "China": 0.18,        # 18%
    "India": 0.15,        # 15%
    "Brazil": 0.12,       # 12%
    "USA": 0.08,          # 8%
    "UK": 0.06,           # 6%
    "Germany": 0.05,      # 5%
    "France": 0.04,       # 4%
    "Canada": 0.03        # 3%
}
# Location:
locations = ["USA", "UK", "China", "Japan", "France", "Germany"]
# Define location weights (if you want some locations to appear more frequently)
location_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Adjust these weights as needed

# Devices and device weights
devices = ["iPhone", "Android", "Windows Laptop", "MacBook", "Linux PC", "iPad"]
device_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]  # Weights representing how frequently each device is used

# Networks and network weights
networks = ["Home WiFi", "Public WiFi", "Mobile Network", "Corporate Network"]
network_weights = [0.4, 0.3, 0.2, 0.1]  # Weights indicating the frequency of each network


def check_and_normalize_catalog(behavior_catalog, output_file="src/normalized_catalog"):
    #os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    """
    Function to check if each row of the transition probability matrix sums to one 
    to prevent floating point issue in the sequence extraction.
    The normalized catalog is saved as "normalized_catalog.json"
    """
    normalized_catalog = {}
    
    for behavior, data in behavior_catalog.items():
        activities = data["activities"]
        matrix = np.array(data["transition_matrix"])  

        # Normalize each row
        for i in range(matrix.shape[0]):
            row_sum = np.sum(matrix[i])
            if row_sum != 0:
                matrix[i] = matrix[i] / row_sum

        # Adjust for floating-point precision issues
        for i in range(matrix.shape[0]):
            row_sum = np.sum(matrix[i])
            if abs(row_sum - 1) > 1e-6:
                matrix[i, -1] += 1 - row_sum

        normalized_catalog[behavior] = {
            "activities": activities,
            "transition_matrix": matrix.tolist(),
            "time_limit": data["time_limit"],
            "fraud": data["fraud"]
        }

        # Save each behavior matrix as a JSON file
        # Ensure output directory exists
        #output_dir = "normalized_catalog"
        #output_file = os.path.join(output_dir, f"{behavior}.json")
        #print(normalized_catalog[behavior])
    with open(f'{output_file}.json', "w") as f:
        json.dump(normalized_catalog, f, indent=4)

    print(f"Normalized transition matrices saved in {output_file}.json")
    return normalized_catalog


'''
TRANSACTION_MERCHANTS = {
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
'''


'''
# Define behavior activity_types and their corresponding activities
behavior_activities = {
    "normal": [
        "Open Account", "Deposit Funds", "Make Purchase", "Pay Bills", 
        "Transfer Funds", "Apply for Credit Card", "Apply for Loan", 
        "Invest Money", "Review activitiyments", "Close Account"
    ],
    "identity_theft": [
        "Failed Login", "Suspicious Login", "Change Password", "Change Email", 
        "Change Phone", "Request New Card", "Unauthorized Transfer", "Close Account"
    ],
    "card_skimming": [
        "ATM Withdrawal", "Online Purchase", "POS Purchase", "Balance Check", 
        "Request New PIN", "Report Lost Card", "Close Account"
    ]
}

# This code block can be used to create a catalog with random transition probabilities
# Generate random transition matrices for each behavior. This use Dirichlet distribution to ensure that each row sum to one
def generate_transition_matrix(activities):
    num_activities = len(activities)
    matrix = np.random.dirichlet(np.ones(num_activities), size=num_activities)  # Ensuring each row sums to 1
    return matrix.tolist()

# Create the behavior dictionary
behavior_catalog = {
    behavior: {
        "activities": activities,
        "transition_matrix": generate_transition_matrix(activities)
    }
    for behavior, activities in behavior_activities.items()
}
# Save to JSON file
output_file = os.path.join(output_dir, "behavior_catalog.json")
with open(output_file, "w") as f:
    json.dump(behavior_catalog, f, indent=4)

print(f"Behavior catalog saved to {output_file}")
'''
