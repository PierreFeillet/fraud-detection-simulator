import random
import pandas as pd
import numpy as np
import time
import argparse
from pprint import pprint
import os
import json
import random
from datetime import timedelta


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

from IPython import embed
from sklearn.preprocessing import StandardScaler


#from legitimate_agent import LegitimateCustomer
#from fraudulentster_agent import fraudulentster
from activity import Activity
from agent import Agent
from bank import BankActivities
from catalog import behavior_catalog, check_and_normalize_catalog, fraud_rates_by_country, locations,location_weights

def assign_agent_countries(agents):
    """Assign agents to countries based on a predefined distribution."""
    country_distribution = list(fraud_rates_by_country.keys())
    country_weights = [1 / fraud_rates_by_country[c] for c in country_distribution]  # Inverse fraud rate
    total_weight = sum(country_weights)
    country_probabilities = [w / total_weight for w in country_weights]

    agent_countries = {
        agent.agent_id: random.choices(country_distribution, country_probabilities)[0] for agent in agents
    }
    return agent_countries

def run_simulation(fraudster_rate, normalized_catalog, agents, bank, start_time, steps=100, flush_interval=100, target_size=10**6):
    
    activity_time = start_time
    #agent_countries = assign_agent_countries(agents)  # Assign initial countries
    active_agents = {}

    # Initialize active agents
    for agent in agents:
        #country = agent_countries[agent.agent_id]
        #fraud_probability = fraud_rates_by_country[country]
        is_fraudulent = random.random() < fraudster_rate
        behavior_type = get_random_fraud_behavior(normalized_catalog) if is_fraudulent else "legitimate"

        active_agents[agent.agent_id] = {
            "behavior": behavior_type,
            #"residence_country": agent.residence_country, #will be used to extract probabilities of an activity made in a certain location
            "balance": agent.initial_balance,
            "last_activity": None
        }

    for step in range(steps):
        if len(bank.activity_log) >= target_size:
            print(f"Target generation of {target_size} activities reached")
            break

        for agent in list(agents):  # Iterate over agents
            if agent.agent_id not in active_agents:
                continue  # Skip closed accounts

            behavior_type = active_agents[agent.agent_id]["behavior"]
            #country = active_agents[agent.agent_id]["country"]
            #fraud_status = active_agents[agent.agent_id]["fraud"]

            behavior = normalized_catalog[behavior_type]
            activities = behavior["activities"]
            time_limit = behavior["time_limit"]
            transition_matrix = behavior["transition_matrix"]

            # Select an initial activity
            if active_agents[agent.agent_id]["last_activity"] is None:
                valid_activities = [act for act in activities.keys() if act != "Close Account"]
                current_activity_type = random.choice(valid_activities) if valid_activities else None
            else:
                current_activity_type = extract_activity_markov_chain(
                    active_agents[agent.agent_id]["last_activity"], activities, transition_matrix
                )

            if current_activity_type is None:
                continue  

            delta = timedelta(minutes=random.randint(0, time_limit))
            activity_time += delta
            current_activity = Activity(
                agent_id=agent.agent_id,
                initial_balance=active_agents[agent.agent_id]["balance"],
                timestamp=activity_time.strftime("%Y-%m-%d %H:%M:%S"),
                behavior=behavior_type,
                residence_country=agent.residence_country
            )
            current_activity.activity_type = current_activity_type
            transaction_range = activities[current_activity_type]
            current_activity.amount = 0 if transaction_range == (0, 0) else random.randint(*transaction_range)
            current_activity.update_balance()
            current_activity.fraud = normalized_catalog[behavior_type]['fraud']

            active_agents[agent.agent_id]["balance"] = current_activity.balance
            active_agents[agent.agent_id]["last_activity"] = current_activity_type
            bank.add_activity(current_activity)

            # If the agent closes their account, remove them from active agents
            if current_activity_type == "Close Account":
                del active_agents[agent.agent_id]

        # Check if the buffer size has reached 100 and flush if necessary
        if len(bank.buffer) >= flush_interval:
            bank.flush_activities()
            print(f"Flushed transactions at step {step}")
    
    bank.last_activity_time = activity_time
    # Final flush after simulation ends
    bank.flush_activities()
    print(f"Final flush completed.")




def get_random_fraud_behavior(normalized_catalog):
    fraud_behaviors = [key for key in normalized_catalog.keys() if key != "legitimate"]
    return random.choice(fraud_behaviors)

def extract_activity_markov_chain(current_activity, activities, transition_matrix):
    current_activity_index = list(activities.keys()).index(current_activity)
    next_activity_type = np.random.choice(list(activities.keys()), p=transition_matrix[current_activity_index])
    return next_activity_type

def extract_features(activity_log):
    X = activity_log[["amount", "balance"]] #"risk_level"
    y = activity_log["fraudulent"]
    return X, y

def benchmark_models(activity_log):
    X, y = extract_features(activity_log)

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

def generate_dataset(fraudster_rate, nb_activities, min_n_agents, data_folder, start_time, target_size):
    initial_agents = [Agent(agent_id=i, initial_balance=round(random.uniform(1000, 5000), 2), residence_country=np.random.choice(locations, p=location_weights)) for i in range(min_n_agents)]
    #fraudulent_agents = [Agent(agent_id=i + nb_legitimate_agents, initial_balance=round(random.uniform(1000, 5000), 2)) for i in range(nb_fraudulent_agents)]
    bank_with_activities = BankActivities(nb_activities)
    run_simulation(fraudster_rate=fraudster_rate, normalized_catalog=normalized_catalog, agents=initial_agents, bank=bank_with_activities, start_time=start_time, target_size=nb_activities)
    last_activity_time = pd.to_datetime(bank_with_activities.last_activity_time)
    # Check if we need to generate new agents
    count = 1
    print(f"Generated number of actitivities: {len(bank_with_activities.activity_log)}, required {target_size}")
    while(len(bank_with_activities.activity_log)<target_size):
        print(f"Generating new agents to reach the target size")
        print(f"Restarting loop over steps..")
        # Get last activity timestamp
        new_agents = [Agent(agent_id=min_n_agents + i, initial_balance=round(random.uniform(1000, 5000), 2), residence_country=np.random.choice(locations, p=location_weights)) for i in range(min_n_agents)]
        run_simulation(fraudster_rate=fraudster_rate,normalized_catalog=normalized_catalog, agents=new_agents, bank=bank_with_activities, start_time=last_activity_time, target_size=nb_activities)
        count = count +1
        print(f"Had to generate {count*min_n_agents} new agents iterating {count} times over the minimum number of agents. \nConsider increasing the minimum number of agents if too many agents are created.")
        print(f"Generated number of actitivities: {len(bank_with_activities.activity_log)}, required {target_size}")

    os.makedirs(data_folder, exist_ok=True)
    file_name= f'fraud_simulation_activities_{format_number(nb_activities)}.csv'
    bank_with_activities.activity_log.to_csv(f"{data_folder}/{file_name}", index=False)
    print(f"Data file saved as {data_folder}/{file_name} ")
    return bank_with_activities.activity_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nb_activities', help='Total number of activities to be generated', type=int, required=True)
    parser.add_argument('--min_n_agents', help='Number of legitimate agents', type=int, default=40)
    parser.add_argument('--fraudster_rate', help='Rate of fraudulent agents', type=float, default=0.1) # Default is High Risk
    #Low-Fraud Environment (Highly Secure Banking System): 0.5% - 2% fraudsters.
    #Moderate Fraud Risk (General Banking, E-commerce): 2% - 5% fraudsters.
    #High-Risk Scenarios (Less Secure Systems, Money Laundering Simulations): 5% - 10% fraudsters.
    parser.add_argument('--data_folder', help='Where to save produced data', type=str, default='data')
    parser.add_argument('--start_time', help='Initial timestamp value for the generating the series (ISO 8601 format, example: "2025-01-06T12:00:00")', default=datetime.now())
    # Example: python src/simulator.py --nb_activities 1000 --n_legitimate_agent 3 --n_fraudulent_agent 1 --pr_fraud 0.3
    cfg = parser.parse_args()
    pprint(cfg)

    # Generate/Read catalog wirh normalized probabilities
    #if os.path.exists('src/normalized_catalog.json'):
    #    with open('src/normalized_catalog.json', "r") as f:
    #        normalized_catalog = json.load(f)
    #else:
    #    normalized_catalog = check_and_normalize_catalog(behavior_catalog)
    normalized_catalog = check_and_normalize_catalog(behavior_catalog) #better to regenerate it everytime in case some probabilitis are changed

    #n_fraudulent_activities = int(cfg.nb_activities*cfg.pr_frauds)
    #n_legitimate_activities = cfg.nb_activities - n_fraudulent_activities
    #n_max_per_legitimate_A = int(n_legitimate_activities/cfg.n_legitimate_agent)
    #n_max_per_fraudster_A = int(n_fraudulent_activities/cfg.n_fraudulent_agent)
    
    dataset = generate_dataset(fraudster_rate=cfg.fraudster_rate, nb_activities=cfg.nb_activities, min_n_agents=cfg.min_n_agents, data_folder=cfg.data_folder, start_time=cfg.start_time, target_size=cfg.nb_activities)
    # Build sample for training ML clustering alghoritms
    #columns_to_drop = ['behavior']
    #dataset.drop(columns=columns_to_drop, inplace=True)
    #
#
    ## Standardize data
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data)
#
    # Benchmark anomaly detection models
    if False:
        results = benchmark_models(activity_log_with_fraudulent_features)
        for model, metrics in results.items():
            print(f"Model: {model}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"Accuracy: {metrics['Accuracy']}")
            print(f"Classification Report: \n{metrics['Classification Report']}")
