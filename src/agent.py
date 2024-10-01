
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

from catalog import behavioral_catalog

class Agent:
    def __init__(self, agent_id, balance):
        self.agent_id = agent_id
        self.balance = balance

class LegitimateCustomer(Agent):
    def generate_transaction(self):
        transaction_type = random.choice(["deposit", "withdrawal"])
        amount = random.uniform(10, 200)
        if transaction_type == "withdrawal" and self.balance > amount:
            self.balance -= amount
        elif transaction_type == "deposit":
            self.balance += amount
        return {"agent_id": self.agent_id, "type": transaction_type, "amount": amount, "balance": self.balance, "fraud": 0}

    def account_activity(self):
        activities = ["password_change", "email_change", "phone_change", "suspicious_login", "failed_login"]
        activity_type = random.choice(activities)
        return {"agent_id": self.agent_id, "type": activity_type, "amount": 0, "balance": self.balance, "fraud": 0}

class Fraudster(Agent):
    def __init__(self, agent_id, balance, fraud_type="identity_theft"):
        super().__init__(agent_id, balance)
        self.fraud_type = fraud_type
        self.behavior = behavioral_catalog[fraud_type]
        self.activity_sequence = self.behavior["activity_behavior"]["sequence"]
        self.time_probabilities = self.behavior["activity_behavior"]["time_probabilities"]

    def commit_fraud(self):
        transaction_type = "withdrawal" if random.random() < self.behavior["transaction_behavior"]["transaction_frequency"] else "deposit"
        if transaction_type == "withdrawal":
            amount = random.uniform(*self.behavior["transaction_behavior"]["withdrawal_range"])
            self.balance -= amount
        else:
            amount = random.uniform(*self.behavior["transaction_behavior"]["deposit_range"])
            self.balance += amount
        return {"agent_id": self.agent_id, "type": "fraud", "amount": amount, "balance": self.balance, "fraud": 1}

    def account_activity(self):
        current_time_step = random.random()
        activity_type = None
        for i, prob in enumerate(self.time_probabilities):
            if current_time_step < prob:
                activity_type = self.activity_sequence[i]
                break
        if not activity_type:
            activity_type = "failed_login"
        return {"agent_id": self.agent_id, "type": activity_type, "amount": 0, "balance": self.balance, "fraud": 1}


