import numpy as np
import pandas as pd
from IPython import embed
# List of activities
normal_activities = [
    "Open Account", "Deposit Funds", "Make Purchase", "Pay Bills", "Transfer Funds",
    "Apply for Credit Card", "Apply for Loan", "Invest Money", "Review activities",
    "Failed Login", "Suspicious Login", "Change Password", "Change Email", "Change Phone",
    "Request New Card", "ATM Withdrawal", "Online Purchase", "POS Purchase", "Balance Check",
    "Request New PIN", "Report Lost Card", "Wire Transfer", "Cash Withdrawal",
    "Invest in Assets", "Max Out Credit", "Withdraw Funds"
]

# Define realistic transition probabilities
num_activities = len(normal_activities)
transition_matrix = np.zeros((num_activities, num_activities))

# Define transition rules for a normal agent
for i, activity in enumerate(normal_activities):
    if activity == "Open Account":
        probs = {"Deposit Funds": 0.4, "Apply for Credit Card": 0.2, "Apply for Loan": 0.2, "Make Purchase": 0.2}
    elif activity == "Deposit Funds":
        probs = {"Make Purchase": 0.4, "Pay Bills": 0.3, "Invest Money": 0.2, "Transfer Funds": 0.1}
    elif activity == "Make Purchase":
        probs = {"Make Purchase": 0.4, "POS Purchase": 0.3, "Online Purchase": 0.2, "Balance Check": 0.1}
    elif activity == "Pay Bills":
        probs = {"Deposit Funds": 0.3, "Transfer Funds": 0.4, "Balance Check": 0.2, "Review activities": 0.1}
    elif activity == "Transfer Funds":
        probs = {"Wire Transfer": 0.3, "Cash Withdrawal": 0.2, "Deposit Funds": 0.2, "Balance Check": 0.3}
    elif activity == "Apply for Credit Card":
        probs = {"Deposit Funds": 0.3, "Make Purchase": 0.3, "Invest Money": 0.2, "Review activities": 0.2}
    elif activity == "Apply for Loan":
        probs = {"Deposit Funds": 0.4, "Invest Money": 0.4, "Review activities": 0.2}
    elif activity == "Invest Money":
        probs = {"Invest in Assets": 0.5, "Withdraw Funds": 0.3, "Review activities": 0.2}
    elif activity == "Review activities":
        probs = {"Balance Check": 0.4, "Make Purchase": 0.3, "Pay Bills": 0.3}
    elif activity in ["Failed Login", "Suspicious Login"]:
        probs = {"Change Password": 0.5, "Review activities": 0.3, "Request New PIN": 0.2}
    elif activity in ["Change Password", "Change Email", "Change Phone"]:
        probs = {"Review activities": 0.5, "Balance Check": 0.3, "Request New PIN": 0.2}
    elif activity == "Request New Card":
        probs = {"Make Purchase": 0.3, "POS Purchase": 0.3, "Online Purchase": 0.3, "Balance Check": 0.1}
    elif activity == "ATM Withdrawal":
        probs = {"Make Purchase": 0.4, "POS Purchase": 0.3, "Balance Check": 0.3}
    elif activity in ["Online Purchase", "POS Purchase"]:
        probs = {"Make Purchase": 0.3, "Balance Check": 0.3, "Review activities": 0.2, "Deposit Funds": 0.2}
    elif activity == "Balance Check":
        probs = {"Make Purchase": 0.3, "Pay Bills": 0.3, "Review activities": 0.4}
    elif activity == "Request New PIN":
        probs = {"Make Purchase": 0.4, "POS Purchase": 0.3, "Online Purchase": 0.3}
    elif activity == "Report Lost Card":
        probs = {"Request New Card": 0.5, "Review activities": 0.5}
    elif activity == "Wire Transfer":
        probs = {"Make Purchase": 0.3, "Invest Money": 0.3, "Withdraw Funds": 0.2, "Balance Check": 0.2}
    elif activity == "Cash Withdrawal":
        probs = {"Make Purchase": 0.4, "POS Purchase": 0.3, "Balance Check": 0.3}
    elif activity == "Invest in Assets":
        probs = {"Withdraw Funds": 0.4, "Invest Money": 0.4, "Balance Check": 0.2}
    elif activity == "Max Out Credit":
        probs = {"Withdraw Funds": 0.5, "Review activities": 0.3, "Balance Check": 0.2}
    elif activity == "Withdraw Funds":
        probs = {"Make Purchase": 0.4, "POS Purchase": 0.3, "Balance Check": 0.3}
    else:
        probs = {activity: 1.0}  # Self-loop if undefined

    # Convert probability dictionary to row vector
    for target_activity, prob in probs.items():
        j = normal_activities.index(target_activity)
        transition_matrix[i, j] = prob

# Convert to DataFrame for better visualization
transition_df = pd.DataFrame(transition_matrix, index=normal_activities, columns=normal_activities)
print(transition_df)
embed()


''' to read file'''
import json

# Load transition matrix from file
with open("transition_matrix.json", "r") as f:
    data = json.load(f)

# Extract activities and transition matrix
activities = data["activities"]
transition_matrix = data["transition_matrix"]

# Define the behavior catalog and integrate the transition matrix
behavior_catalog = {
    "legitimate": {
        "activities": {
            activity: "neutral" if label == "neutral" else ("positive" if label == "positive" else "negative")
            for activity, label in zip(activities, ["neutral", "positive", "negative", "negative", "negative",
                                                    "neutral", "neutral", "negative", "neutral", "neutral",
                                                    "neutral", "neutral", "neutral", "neutral", "neutral",
                                                    "negative", "negative", "negative", "neutral", "neutral",
                                                    "neutral", "negative", "negative", "negative", "negative",
                                                    "negative"])
        },
        "transition_matrix": transition_matrix,
        "time_limit": 180,  # Maximum 3 hours between events
        "fraud": 0,
    }
}

# Save updated behavior catalog to a new file
with open("behavior_catalog.json", "w") as f:
    json.dump(behavior_catalog, f, indent=4)

print("Behavior catalog saved to 'behavior_catalog.json'.")