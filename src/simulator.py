import random
import pandas as pd
import numpy as np
import time
import argparse
from pprint import pprint
import os


from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer

from legitimate_agent import LegitimateCustomer
from fraudster_agent import Fraudster

from catalog import behavioral_catalog, MERCHANT_TRANSACTION_TYPES

# Function to generate a random country
def generate_country():
    return random.choice(COUNTRIES)

class BankActivities:
    def __init__(self, max_size):
        # Define all columns initially, even if empty
        self.dtypes = {
            "agent_id": "int16",
            "timestamp": "datetime64[ns]",
            "type": "category",
            "amount": "float32",
            "balance": "float32",
            "merchant": "category",
            "location": "category",
            "device": "category",
            "network": "category",
            "compromised_device": "int8",
            "compromised_network": "int8",
            "fraud": "int8"}
        
        # Pre-define DataFrame with specified dtypes
        self.transaction_log = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)
        self.current_time = datetime.now()
        self.transactions_buffer = []  # Buffer for temporary transactions
        self.max_size = max_size

    def process_transaction(self, transaction):
        # Check if we are at max size
        if len(self.transaction_log) == self.max_size:
            return  # Stop processing if max size is reached

        transaction["timestamp"] = self.current_time
        self.moveCurrentTime()
        self.transactions_buffer.append(transaction)  # Append to buffer

    def process_activity(self, activity):
        # Check if we are at max size
        if len(self.transaction_log) == self.max_size:
            return  # Stop processing if max size is reached

        activity.setdefault("amount", 0)
        activity["timestamp"] = self.current_time
        self.moveCurrentTime()
        self.transactions_buffer.append(activity)  # Append to buffer

    def moveCurrentTime(self):
        self.current_time += timedelta(seconds=random.randint(1, 30))
        print("Current time now: " + str(self.current_time))

    def flush_transactions(self):
        """Consolidate buffered transactions."""
        """Optimize the code as Pandas concat does not scale in performances."""
        """It now allows a more stable TPS."""
        if self.transactions_buffer:
            try:
                new_data = pd.DataFrame(self.transactions_buffer, columns=self.dtypes.keys()).astype(self.dtypes)

                # Check if adding new_data would exceed max_size
                if len(self.transaction_log) + len(new_data) > self.max_size:
                    # Truncate new_data to fit the remaining space in transaction_log
                    remaining_space = self.max_size - len(self.transaction_log)
                    new_data = new_data.iloc[:remaining_space]

                # Add new data to transaction log
                self.transaction_log = pd.concat([self.transaction_log, new_data], ignore_index=True)
                self.transactions_buffer = []  # Clear buffer

            except pd.errors.OutOfBoundsDatetime as e:
                print("Error:", e)


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

        # Interrupt the generation when the targeted number of activities has been reached
        if len(bank.transaction_log) >= bank.max_size - 1:
            print("Target generation of " + str(bank.max_size) + " actvities reached")
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

def format_number(nb_global_activities):
    if nb_global_activities >= 1_000_000:
        value = nb_global_activities / 1_000_000
        suffix = "M"
    elif nb_global_activities >= 1_000:
        value = nb_global_activities / 1_000
        suffix = "K"
    else:
        return str(nb_global_activities)  # No suffix for numbers less than 1,000

    # Format to remove .0 if the value is an integer
    formated_number = f"{int(value) if value.is_integer() else round(value, 1)}{suffix}"
    return formated_number

def generate_dataset(nb_activities, nb_agents, pr_fraudulent=0.3,):
    # nb_agents sets total number of agents (fraudulent+legitimate)
    # pr_fraudulent sets proportion of fraudulent agent
    nb_fraudster_agents = int(pr_fraudulent*nb_agents)
    nb_legitimate_agents = nb_agents - nb_fraudster_agents
    legitimate_agents = [LegitimateCustomer(agent_id=i, balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_legitimate_agents)]
    fraudster_agents = [Fraudster(agent_id=i+nb_legitimate_agents, balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_fraudster_agents)]
    bank_with_activities = BankActivities(nb_activities)
    run_simulation_with_activities(legitimate_agents, fraudster_agents, bank_with_activities, steps=1000)
    
    transaction_log_with_fraud_features = bank_with_activities.transaction_log
    data_folder = 'data_test'
    os.makedirs(data_folder, exist_ok=True)
    transaction_log_with_fraud_features.to_csv(f"{data_folder}/fraud_simulation_" + format_number(nb_activities) + "_activities.csv", index=False, chunksize=10000)

'''
def test_unitary_agent(nb_activities):
    nb_legitimate_agents = nb_activities // 10
    nb_fraudster_agents = nb_activities // 30
    legitimate_agents = [LegitimateCustomer(agent_id=i, balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_legitimate_agents)]
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nb_activities', help='Total number of activities to be generated', type=int, required=True)
    parser.add_argument('--nb_agents', help='Total number of agents', type=int, required=True)
    parser.add_argument('--pr_fraudulent', help='Proportion of fraudulent agent with respect to the total number of agents', type=float, default=0.3)
    # Example: python src/simulator.py --nb_activities 1000 --nb_agents 10 --pr_fraudulent 0.3

    cfg = parser.parse_args()
    pprint(cfg)

    generate_dataset(cfg.nb_activities, cfg.nb_agents, cfg.pr_fraudulent)

    # Benchmark anomaly detection models
    if False:
        results = benchmark_models(transaction_log_with_fraud_features)
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Accuracy: {metrics['Accuracy']}")
            print(f"Classification Report: \n{metrics['Classification Report']}")
