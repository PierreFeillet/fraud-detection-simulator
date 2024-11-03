import random
import pandas as pd
import numpy as np
import time

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer

from agent import LegitimateCustomer, Fraudster
from catalog import behavioral_catalog

class BankWithClientActivities:
    def __init__(self, max_size):
        # Define all columns initially, even if empty
        self.columns = ["agent_id", "type", "amount", "balance", "timestamp", "fraud", "risk_level"]
        self.transaction_log = pd.DataFrame(columns=self.columns)
        self.current_time = datetime.now()
        self.transactions_buffer = []  # Buffer for temporary transactions
        self.max_size = max_size  # Set maximum size for transaction log

    def process_transaction(self, transaction):
        # Check if we are at max size
        if len(self.transaction_log) >= self.max_size - 1:
            return  # Stop processing if max size is reached

        transaction["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transactions_buffer.append(transaction)  # Append to buffer

    def process_activity(self, activity):
        # Check if we are at max size
        if len(self.transaction_log) >= self.max_size - 1:
            return  # Stop processing if max size is reached

        activity.setdefault("amount", 0)
        activity["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transactions_buffer.append(activity)  # Append to buffer

    def flush_transactions(self):
        """Consolidate buffered transactions."""
        """Optimize the code as Panda concat does not scale in performances."""
        """It now allows a stable TPS."""
        if self.transactions_buffer:
            new_data = pd.DataFrame(self.transactions_buffer, columns=self.columns)

            # Check if adding new_data would exceed max_size
            if len(self.transaction_log) + len(new_data) > self.max_size - 1:
                # Truncate new_data to fit the remaining space in transaction_log
                remaining_space = self.max_size - len(self.transaction_log) - 1
                new_data = new_data.iloc[:remaining_space]

            # Add new data to transaction log
            self.transaction_log = pd.concat([self.transaction_log, new_data], ignore_index=True)
            self.transactions_buffer = []  # Clear buffer


def run_simulation_with_activities(legitimate_agents, fraudster_agents, bank, steps=100, flush_interval=100):
    groupFactor = 10
    start_time = time.time()
    for step in range(steps):
        if step == 0:
            print(f"step: {step}")
            start_time = time.time()

        if step != 0 and step % groupFactor == 0:
            end_time = time.time()
            print(f"step: {step}")

            # Calculate the duration
            duration = end_time - start_time
            tps = groupFactor / duration
            print(f"TPS: {tps}")
            start_time = time.time()

        # Only continue if the transaction log is not full
        if len(bank.transaction_log) >= bank.max_size - 1:
            print("Reached maximum transaction log size.")
            break

        for agent in legitimate_agents:
            transaction = agent.generate_transaction()
            bank.process_transaction(transaction)
            if random.random() < 0.2:
                activity = agent.account_activity()
                bank.process_activity(activity)

        for fraudster in fraudster_agents:
            fraud_type = random.choice(list(behavioral_catalog.keys()))
            fraud = fraudster.commit_fraud(fraud_type)
            bank.process_transaction(fraud)
            if random.random() < 0.5:
                activity = fraudster.account_activity(fraud_type)
                bank.process_activity(activity)

        # Flush transactions to the DataFrame every `flush_interval` steps
        if step % flush_interval == 0:
            bank.flush_transactions()

    # Final flush after the loop
    bank.flush_transactions()

def extract_features(transaction_log):
    X = transaction_log[["amount", "balance"]] #"risk_level"
    y = transaction_log["fraud"]
    return X, y

def benchmark_models(transaction_log):
    X, y = extract_features(transaction_log)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")  # Replace NaNs with the mean of the column
    X = imputer.fit_transform(X)

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
    nb_global_activities = 100000
    nb_legitimate_agents = nb_global_activities // 10
    nb_fraudster_agents = nb_global_activities // 30
    legitimate_agents = [LegitimateCustomer(agent_id=i, balance=random.uniform(1000, 5000)) for i in range(nb_legitimate_agents)]
    fraudster_agents = [Fraudster(agent_id=i+100, balance=random.uniform(1000, 5000)) for i in range(nb_fraudster_agents)]
    bank_with_activities = BankWithClientActivities(nb_global_activities)
    run_simulation_with_activities(legitimate_agents, fraudster_agents, bank_with_activities, steps=1000)
    
    transaction_log_with_fraud_features = bank_with_activities.transaction_log
    transaction_log_with_fraud_features.to_csv("data/fraud_simulation_" + str(nb_global_activities) + "_activities.csv", index=False, chunksize=10000)

    # Benchmark anomaly detection models
    if False:
        results = benchmark_models(transaction_log_with_fraud_features)
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Accuracy: {metrics['Accuracy']}")
            print(f"Classification Report: \n{metrics['Classification Report']}")
