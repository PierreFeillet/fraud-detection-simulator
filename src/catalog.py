
# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Behavioral catalog for various fraudulent behaviors with time series and activity sequences
behavioral_catalog = {
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