import random
import pandas as pd
import numpy as np
import time
import argparse
from pprint import pprint
import os
import json


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

from IPython import embed

#from legitimate_agent import LegitimateCustomer
#from fraudulentster_agent import fraudulentster
from operation import Operation
from agent import Agent
from bank import BankActivities
from catalog import behavior_catalog, check_and_normalize_catalog
# Function to generate a random country
def generate_country():
    return random.choice(COUNTRIES)

def run_simulation_with_activities(normalized_catalog, legitimate_agents, fraudulent_agents, bank, start_time, steps=100, flush_interval=100):
    start_time = time.time()
    for step in range(steps):
        if len(bank.operation_log) == bank.max_size:
            print(f"Target generation of {bank.max_size} activities reached")
            break

        # The following two for block can be unified if
        for agent in legitimate_agents:
            activity_sequence = simulate_markov_chain(normalized_catalog=normalized_catalog, start_time=start_time, behavior_type='legitimate',)
            df_sequence = pd.DataFrame(activity_sequence)
            agent_operations = generate_agent_operations(agent, df_sequence)
            print(agent_operations)
            bank.add_operations(agent_operations)

        for agent in fraudulent_agents:
            fraudulent_behavior = get_random_fraud_behavior(normalized_catalog)
            agent_operations = generate_agent_operations(agent.agent_id, behavior_type=fraudulent_behavior, timestamp=start_time,)
            bank.add_operations(agent_operations)

        if step % flush_interval == 0:
            bank.flush_transactions()
            print(f"Flushed transactions at step {step}")

    bank.flush_transactions()

def get_random_fraud_behavior(normalized_catalog):
    fraud_behaviors = [key for key in normalized_catalog.keys() if key != "legitimate"]
    return random.choice(fraud_behaviors)

def simulate_markov_chain(normalized_catalog, start_time, behavior_type, n_activitiy=20):
    activities = normalized_catalog[behavior_type]["activities"]
    transition_matrix = normalized_catalog[behavior_type]["transition_matrix"]
    time_limit = normalized_catalog[behavior_type]["time_limit"]
    activity_sequence = []

    current_activity = random.choice(list(activities.keys()))
    transaction_range = activities[current_activity]
    transaction_amount = 0 if transaction_range == (0, 0) else random.randint(*transaction_range)
    timestamp = datetime.fromtimestamp(start_time)
    
    # Add the first activity to the sequence
    activity_sequence.append({
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "activity": current_activity,
        "amount": transaction_amount
    })

    for _ in range(n_activitiy):
        current_activity_index = list(activities.keys()).index(current_activity)
        next_activity = np.random.choice(list(activities.keys()), p=transition_matrix[current_activity_index])
        transaction_range = activities[next_activity]
        transaction_amount = 0 if transaction_range == (0, 0) else random.randint(*transaction_range)
        timestamp += timedelta(minutes=random.randint(1, time_limit * 60))

        # Add the next activity to the sequence
        activity_sequence.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "activity": str(next_activity),
            "amount": transaction_amount
        })

        if next_activity == "Close Account":
            break
        
        current_activity = next_activity

    return activity_sequence



def generate_agent_operations(agent_id, df):
    operations = []
    balance = round(random.uniform(1000, 5000), 2)
    
    # Iterate over the DataFrame, which contains the operations
    for i, row in df.iterrows():
        # Get timestamp and amount for the current operation
        timestamp = pd.to_datetime(row['timestamp'])
        amount = row['amount']
        
        # Create Operation instance with the given timestamp and amount
        operation = Operation(
            agent_id=agent_id, 
            initial_balance=balance,
            timestamp=timestamp,
        )
        operation.amount = amount
        # Update balance based on the operation
        operation.update_balance()
       # operation.extract_merchant()
        
        # Record operation details
        operation_data = {key: getattr(operation, key) for key in Operation.OPERATION_DTYPE.keys()}
        operations.append(operation_data)
        # Update balance for the next operation
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

def generate_dataset(nb_activities, nb_agents, data_folder, start_time, pr_fraudulent=0.3):
    nb_fraudulent_agents = int(pr_fraudulent * nb_agents)
    nb_legitimate_agents = nb_agents - nb_fraudulent_agents

    legitimate_agents = [Agent(agent_id=i, initial_balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_legitimate_agents)]
    fraudulent_agents = [Agent(agent_id=i + nb_legitimate_agents, initial_balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_fraudulent_agents)]

    bank_with_activities = BankActivities(nb_activities)
    run_simulation_with_activities(normalized_catalog=normalized_catalog, legitimate_agents=legitimate_agents, fraudulent_agents=fraudulent_agents, bank=bank_with_activities, start_time=start_time, steps=1000)
    
    operation_log_with_fraudulent_features = bank_with_activities.operation_log
    os.makedirs(data_folder, exist_ok=True)
    operation_log_with_fraudulent_features.to_csv(f"{data_folder}/fraudulent_simulation_{format_number(nb_activities)}_activities.csv", index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nb_activities', help='Total number of activities to be generated', type=int, required=True)
    parser.add_argument('--n_legitimate_agent', help='Number of legitimate agents', type=int, default=2)
    parser.add_argument('--n_fraudulent_agent', help='Number of fraudulent agents', type=int, default=2)
    parser.add_argument('--pr_frauds', help='Percentage of frauds wanted in the dataset', type=float, default=0.01)
    parser.add_argument('--data_folder', help='Where to save produced data', type=str, default='data')
    parser.add_argument('--start_time', help='Initial timestamp value for the generating the series (ISO 8601 format, example: "2025-01-06T12:00:00")', default=datetime.now())
    # Example: python src/simulator.py --nb_activities 1000 --nb_agents 10 --pr_fraudulent 0.3
    cfg = parser.parse_args()
    pprint(cfg)

    # Generate/Read catalog wirh normalized probabilities
    if os.path.exists('src/normalized_catalog.json'):
        with open('src/normalized_catalog.json', "r") as f:
            normalized_catalog = json.load(f)
    else:
        normalized_catalog = check_and_normalize_catalog(behavior_catalog)

    n_fraudulent_activities = int(cfg.nb_activities*cfg.pr_frauds)
    n_legitimate_activities = cfg.nb_activities - n_fraudulent_activities
    n_max_per_legitimate_A = int(n_legitimate_activities/cfg.n_legitimate_agent)
    n_max_per_fraudster_A = int(n_fraudulent_activities/cfg.n_fraudulent_agent)
    embed()
    operation_log_with_fraudulent_features = generate_dataset(nb_activities=cfg.nb_activities, nb_agents=cfg.nb_agents, data_folder=cfg.data_folder, pr_frauds=cfg.pr_frauds, start_time=cfg.start_time)
    

    # Benchmark anomaly detection models
    if False:
        results = benchmark_models(operation_log_with_fraudulent_features)
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Accuracy: {metrics['Accuracy']}")
            print(f"Classification Report: \n{metrics['Classification Report']}")
