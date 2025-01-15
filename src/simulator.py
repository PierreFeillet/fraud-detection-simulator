import random
import pandas as pd
import numpy as np
import time
import argparse
from pprint import pprint
import os


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer

#from legitimate_agent import LegitimateCustomer
#from fraudulentster_agent import fraudulentster
from operation import Operation
from agent import Agent
from bank import BankActivities
from catalog import behavioral_catalog

# Function to generate a random country
def generate_country():
    return random.choice(COUNTRIES)


def run_simulation_with_activities(legitimate_agents, fraudulent_agents, behavioral_catalog, bank, timestamp, steps=100, flush_interval=100):
    start_time = time.time()
    for step in range(steps):
        if len(bank.operation_log) == bank.max_size:
            print(f"Target generation of {bank.max_size} activities reached")
            break

        # The following two for block can be unified if
        for agent in legitimate_agents:
            legitimate_behavior = random.choice([key for key, value in behavioral_catalog.items() if value["fraudulent"] == 0])
            agent_operations = generate_agent_operations(agent.agent_id, behavior_type=legitimate_behavior, timestamp=timestamp)
            bank.add_operations(agent_operations)

        for agent in fraudulent_agents:
            fraudulent_behavior = random.choice([key for key, value in behavioral_catalog.items() if value["fraudulent"] == 1])
            agent_operations = generate_agent_operations(agent.agent_id, behavior_type=fraudulent_behavior, timestamp=timestamp)
            bank.add_operations(agent_operations)

        if step % flush_interval == 0:
            bank.flush_transactions()
            print(f"Flushed transactions at step {step}")

    bank.flush_transactions()

def generate_agent_operations(agent_id, behavior_type, timestamp, max_operations=10):
    num_operations = random.randint(1, max_operations)
    operations = []
    balance = round(random.uniform(1000, 5000), 2)
    for _ in range(num_operations):
        operation = Operation(agent_id=agent_id, initial_balance=balance, behavior_type=behavior_type, timestamp=timestamp)
        operation.moveTime()
        # Update the agent' balance based on the transaction
        operation.initial_balance = balance
        if operation.granted:
            operation.update_balance()
        operation_data = {key: getattr(operation, key) for key in Operation.OPERATION_DTYPE.keys()}
        operations.append(operation_data)
        balance = operation.balance
    return operations

def extract_features(operation_log):
    X = operation_log[["amount", "balance"]] #"risk_level"
    y = operation_log["fraudulent"]
    return X, y

def benchmark_models(operation_log):
    X, y = extract_features(operation_log)

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

def generate_dataset(nb_activities, nb_agents, data_folder, timestamp, pr_fraudulent=0.3):
    nb_fraudulent_agents = int(pr_fraudulent * nb_agents)
    nb_legitimate_agents = nb_agents - nb_fraudulent_agents

    legitimate_agents = [Agent(agent_id=i, initial_balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_legitimate_agents)]
    fraudulent_agents = [Agent(agent_id=i + nb_legitimate_agents, initial_balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_fraudulent_agents)]

    bank_with_activities = BankActivities(nb_activities)
    run_simulation_with_activities(legitimate_agents, fraudulent_agents, behavioral_catalog, bank_with_activities, timestamp, steps=1000)

    operation_log_with_fraudulent_features = bank_with_activities.operation_log
    os.makedirs(data_folder, exist_ok=True)
    operation_log_with_fraudulent_features.to_csv(f"{data_folder}/fraudulent_simulation_{format_number(nb_activities)}_activities.csv", index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nb_activities', help='Total number of activities to be generated', type=int, required=True)
    parser.add_argument('--nb_agents', help='Total number of agents', type=int, required=True)
    parser.add_argument('--pr_fraudulent', help='Proportion of fraudulentulent agent with respect to the total number of agents', type=float, default=0.3)
    parser.add_argument('--data_folder', help='Where to save produced data', type=str, default='data')
    parser.add_argument('--initial_timestamp', help='Initial timestamp value for the generating the series (ISO 8601 format, example: "2025-01-06T12:00:00")', default=None)
    # Example: python src/simulator.py --nb_activities 1000 --nb_agents 10 --pr_fraudulent 0.3

    cfg = parser.parse_args()
    pprint(cfg)

    operation_log_with_fraudulent_features = generate_dataset(nb_activities=cfg.nb_activities, nb_agents=cfg.nb_agents, data_folder=cfg.data_folder, pr_fraudulent=cfg.pr_fraudulent, timestamp=cfg.initial_timestamp)

    # Benchmark anomaly detection models
    if False:
        results = benchmark_models(operation_log_with_fraudulent_features)
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Accuracy: {metrics['Accuracy']}")
            print(f"Classification Report: \n{metrics['Classification Report']}")
