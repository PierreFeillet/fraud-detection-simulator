
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

class BankWithClientActivities:
    def __init__(self):
        self.transaction_log = pd.DataFrame(columns=["agent_id", "type", "amount", "balance", "timestamp", "fraud", "risk_level"])
        self.current_time = datetime.now()

    def process_transaction(self, transaction):
        transaction["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transaction_log = pd.concat([self.transaction_log, pd.DataFrame([transaction])], ignore_index=True)

    def process_activity(self, activity):
        activity["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transaction_log = pd.concat([self.transaction_log, pd.DataFrame([activity])], ignore_index=True)

def run_simulation_with_activities(legitimate_agents, fraudster_agents, bank, steps=100):
    for _ in range(steps):
        for agent in legitimate_agents:
            transaction = agent.generate_transaction()
            bank.process_transaction(transaction)
            if random.random() < 0.2:
                activity = agent.account_activity()
                bank.process_activity(activity)

        for fraudster in fraudster_agents:
            fraud = fraudster.commit_fraud()
            bank.process_transaction(fraud)
            if random.random() < 0.5:
                activity = fraudster.account_activity()
                bank.process_activity(activity)

def benchmark_models(transaction_log):
    X, y = extract_features(transaction_log)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    models = {
        "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True),
        "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    }
    results = {}
    for model_name, model in models.items():
        if model_name == "Local Outlier Factor":
            model.fit(X_train)
            y_pred_test = model.predict(X_test)
        else:
            model.fit(X_train)
            y_pred_test = model.predict(X_test)
        y_pred_test = np.where(y_pred_test == -1, 1, 0)
        results[model_name] = {
            "F1 Score": f1_score(y_test, y_pred_test),
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Classification Report": classification_report(y_test, y_pred_test)
        }
    return results

if __name__ == "__main__":
    legitimate_agents = [LegitimateCustomer(agent_id=i, balance=random.uniform(1000, 5000)) for i in range(3)]
    fraudster_agents = [Fraudster(agent_id=i+3, balance=random.uniform(1000, 5000)) for i in range(1)]
    bank_with_activities = BankWithClientActivities()
    run_simulation_with_activities(legitimate_agents, fraudster_agents, bank_with_activities, steps=100)
    transaction_log_with_fraud_features = bank_with_activities.transaction_log
    transaction_log_with_fraud_features.to_csv("data/fraud_simulation_100_activities.csv", index=False)
