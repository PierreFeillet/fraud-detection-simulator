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

import ollama

from activity import Activity
from agent import Agent
from bank import BankActivities
from catalog import behavior_catalog, check_and_normalize_catalog, fraud_rates_by_country, locations,location_weights
from distributions import TransactionDistributions

def assign_agent_countries(agents):
    """Assign agents to countries based on a predefined distribution."""
    country_distribution = list(fraud_rates_by_country.keys())
    country_weights = [1 / fraud_rates_by_country[c] for c in country_distribution]  # Inverse fraud rate
    total_weight = sum(country_weights)
    country_probabilities = [w / total_weight for w in country_weights]

    agent_countries = {
        agent.real_id: random.choices(country_distribution, country_probabilities)[0] for agent in agents
    }
    return agent_countries

def get_random_fraudster_behavior(normalized_catalog):
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

def assign_behavior(fraudster_rate, normalized_catalog, exclude_identity_theft=False):
    is_fraudulent = random.random() < fraudster_rate

    if is_fraudulent:
        # Get all fraud behaviors excluding 'identity_theft' if needed
        fraud_behaviors = [b for b, v in normalized_catalog.items() if v.get("fraud", 0) == 1]
        if exclude_identity_theft:
            fraud_behaviors = [b for b in fraud_behaviors if b != "identity_theft"]

        behavior_type = random.choice(fraud_behaviors) if fraud_behaviors else "legitimate"
    else:
        behavior_type = "legitimate"

    return is_fraudulent, behavior_type

def generate_new_agents(min_index, max_index, agents, active_agents, normalized_catalog, fraudster_rate, start_time,):
    print(f"Generating new agents: min_index={min_index}, max_index={max_index}")
    
    for i in range(min_index, max_index):
        real_id = i
        
        is_fraudster, behavior_type = assign_behavior(fraudster_rate, normalized_catalog)
        
        # Ensure the first agent in the batch is never an identity thief (to prevent a deadlock)
        while real_id == min_index and behavior_type == 'identity_theft':
            is_fraudster, behavior_type = assign_behavior(fraudster_rate, normalized_catalog)

        # Identity Theft needs a legitimate victim
        if behavior_type == 'identity_theft':
            legitimate_agents = [a for a in agents if not a.is_fraudster and a.real_id in active_agents]  # Find all legit agents
            if legitimate_agents:
                victim_agent = random.choice(legitimate_agents)  # Pick a random legitimate victim
                virtual_id = victim_agent.real_id  # Use the victim's real_id
            else:
                # If no legitimate agents exist, switch to another fraud behavior
                print(f"No legitimate agents available. Assigning different fraud behavior to agent {real_id}.")
                is_fraudster, behavior_type = assign_behavior(fraudster_rate, normalized_catalog, exclude_identity_theft=True)
                virtual_id = real_id  # Default to self ID for non-identity-theft behaviors
        else:
            virtual_id = real_id  # Legitimate or other fraud types use their own ID
        
        # Create and add the agent to the list
        initial_start_time = start_time+timedelta(seconds=random.randint(0, 60))
        agent = Agent(real_id, virtual_id, is_fraudster, behavior_type, initial_start_time)
        agents.append(agent)
        active_agents[real_id] = {
            "balance": agent.initial_balance,
            "last_activity": None,
            "time": agent.initial_time
        }

    return agents



def get_next_activity_from_ollama(agent_data, previous_activity, previous_timestamp):
    """Get the next activity prediction from Ollama."""
    
    # Construct the LLM prompt for Ollama
    prompt = (
        f"You are a structured data generator. You must always return a response in strict JSON format, without any additional text. "
        f"Given the following agent details, predict their next financial activity:\n\n"
        f"Agent Type: {agent_data['agent_type']} (Options: 'traveler', 'static')\n"
        f"Residence Country: {agent_data['initial_country']}\n"
        f"Behavior Type: {agent_data['behavior']} (Examples: 'high spender', 'frequent traveler', 'identity thief', 'card skimmer', 'normal user')\n"
        f"Is Fraudster: {agent_data['is_fraudster']} (0 = Legitimate, 1 = Fraudster)\n"
        f"Previous Activity: {previous_activity} (Options: 'withdrawal', 'deposit', 'purchase', 'login', 'account takeover', etc.)\n"
        f"Previous Timestamp: {previous_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Previous Location: {agent_data['previous_location']}\n"
        f"Previous Amount: {agent_data['previous_amount']} (Use 0 for non-transaction activities)\n\n"
        f"Constraints:\n"
        f"- The predicted activity must align with the agent's **behavior**.\n"
        f"- Timestamp must be after the previous timestamp, with a delay between **5 minutes and 2 hours**.\n"
        f"- The activity must be consistent with past behavior and agent type.\n"
        f"- If the activity is a **transaction**, predict a reasonable amount based on behavior (e.g., small purchases <$100, large withdrawals >$500).\n"
        f"- If the agent is a **fraudster**, they are more likely to conduct suspicious activities such as high-value transactions, international transfers, or account takeovers.\n"
        f"- Return **only** JSON format, like this:\n"
        f"  {{\"activity\": \"...\", \"timestamp\": \"YYYY-MM-DD HH:MM:SS\", \"location\": \"...\", \"amount\": ...}}\n"
    )

    # Send the request to Ollama
    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an AI trained to return structured JSON outputs. Always return valid JSON, nothing else."},
        {"role": "user", "content": prompt}
    ])
    
    # Parse the response and return the result
    try:
        data = json.loads(response['message']['content'])
        return data
    except Exception as e:
        print("Error parsing response from Ollama:", str(e))
        return None


def run_simulation_step(active_agents, normalized_catalog, agents, bank, distributions, steps=100, flush_interval=100, target_size=10**6):
    for step in range(steps):
        if len(bank.activity_log) >= target_size:
            print(f"Target generation of {target_size} activities reached")
            break

        for agent in agents:  # Iterate over agents
            if agent.real_id not in active_agents:
                continue  # Skip closed accounts

            behavior = normalized_catalog[agent.behavior]  # Behavior catalog
            activities = behavior["activities"]

            # Select an initial activity
            if active_agents[agent.real_id]["last_activity"] is None:
                valid_activities = [act for act in activities.keys() if act != "Close Account"]
                current_activity_type = random.choice(valid_activities)
            else:
                # Get the next activity from Ollama (replacing your markov chain logic)
                agent_data = {
                    "agent_type": agent.behavior,
                    "initial_country": active_agents[agent.real_id]["initial_country"],  # Now dynamically populated
                    "behavior": agent.behavior,
                    "is_fraudster": agent.is_fraudster,
                    "previous_activity": active_agents[agent.real_id]["last_activity"],
                    "previous_timestamp": active_agents[agent.real_id]["time"],
                    "previous_location": active_agents[agent.real_id]["previous_location"],  # Now dynamically populated
                    "previous_amount": active_agents[agent.real_id]["previous_amount"],  # Now dynamically populated
                }
                response_data = get_next_activity_from_ollama(agent_data, active_agents[agent.real_id]["last_activity"], active_agents[agent.real_id]["time"])
                if response_data:
                    current_activity_type = response_data["activity"]
                    print(f"Predicted activity for agent {agent.real_id}: {current_activity_type}")

            # Create the activity object and update agent balances and other data
            current_activity = Activity(agent=agent)
            current_activity.initial_balance = active_agents[agent.virtual_id]["balance"]
            current_activity.balance = active_agents[agent.virtual_id]["balance"]
            current_activity.activity_type = current_activity_type
            current_activity.timestamp = activity_time.strftime("%Y-%m-%d %H:%M:%S")

            # Update agent data
            active_agents[agent.real_id]["balance"] = current_activity.balance
            active_agents[agent.real_id]["time"] = pd.to_datetime(current_activity.timestamp)
            active_agents[agent.real_id]["previous_location"] = current_activity.location
            active_agents[agent.real_id]["previous_amount"] = current_activity.amount
            active_agents[agent.real_id]["last_activity"] = current_activity_type
            bank.add_activity(current_activity)

            # Handle account closure
            if current_activity_type == "Close Account":
                closed_id = agent.real_id
                identity_theft_agents = [aid for aid, data in active_agents.items() if data.get("virtual_id") == closed_id]
                for identity_thief in identity_theft_agents:
                    del active_agents[identity_thief]
                    print(f"Identity theft agent {identity_thief} removed due to victim account closure.")
                del active_agents[closed_id]
                print(f"Agent {closed_id} closed their account and was removed.")

        if len(bank.buffer) >= flush_interval:
            bank.flush_activities()
            print(f"Flushed transactions at step {step}")

    bank.flush_activities()



    

def run_full_simulation(normalized_catalog, fraudster_rate, min_n_agents, distributions, start_time, nb_activities, target_size, data_folder):
    count = 0
    bank_with_activities = BankActivities(nb_activities)
    
    # Generate initial agents
    #initial_agents = generate_new_agents(count, min_n_agents, normalized_catalog, fraudster_rate, locations, location_weights)
    agents=[]   
    active_agents = {}
    while len(bank_with_activities.activity_log) < target_size:
        min_index = count * min_n_agents
        max_index = (count + 1) * min_n_agents
        print(f"Generating activities to reach the target size. Iteration: {count}")
        
        # Generate new agents for this batch
        agents = generate_new_agents(min_index, max_index, agents, active_agents, normalized_catalog, fraudster_rate, start_time)

        
        # Run simulation step for the new agents
        run_simulation_step(active_agents,normalized_catalog, agents[min_index:max_index], bank_with_activities, distributions, target_size=nb_activities)
        
        # Update the last activity time after running the simulation
        #last_activity_time = pd.to_datetime(bank_with_activities.last_activity_time)
        count += 1
        print(f"Generated number of activities: {len(bank_with_activities.activity_log)}, required {target_size}")

    print(f"Simulation completed. Total generated activities: {len(bank_with_activities.activity_log)}")

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
    parser.add_argument('--min_n_agents', help='Number of legitimate agents', type=int, default=4)
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
    distributions = TransactionDistributions()
    distributions.generate()
    # Plot log-norm distributions used for extracting transaction amount
    #distributions.plot_distributions()

    dataset = run_full_simulation(fraudster_rate=cfg.fraudster_rate, nb_activities=cfg.nb_activities, min_n_agents=cfg.min_n_agents, normalized_catalog=normalized_catalog, distributions=distributions, data_folder=cfg.data_folder, start_time=cfg.start_time, target_size=cfg.nb_activities)
    #dataset = generate_dataset(fraudster_rate=cfg.fraudster_rate, normalized_catalog=normalized_catalog, nb_activities=cfg.nb_activities, min_n_agents=cfg.min_n_agents, data_folder=cfg.data_folder, start_time=cfg.start_time, target_size=cfg.nb_activities)
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
