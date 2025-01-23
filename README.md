
# Fraud Detection Simulator

## Overview
This project simulates both legitimate and fraudulent transactions in a banking system in a multi-agent architecture. The simulation includes geographical information, merchants, devices, networks, and analyzes whether devices or networks are compromised.

Nominal and fraudulent behaviors are modeled in a catalog as Markov chains. Each chain is a time sequence of activities with an associated probability.
Each beahviour is described through a descriptor in the - [catalog](catalog.py)
All activities participating to nominal and fraudulent behavioral sequences are meshed into a unique dataset and timeline.

## Models for fraudulent and normal behaviors
behaviors are captured in the catalog file and represent several known patterns of normal or fraudulent activities:
* normal
* identity_theft
* money_laundering
* phishing
* card_skimming

They capture a simplified sequence of activities and can be endlessly improved to better refect state of real activities and fraud evolutions.

## What is a fraud?
Fraud in a banking system refers to any intentional deception or misrepresentation carried out by individuals or entities to gain unauthorized access to financial resources, manipulate transactions, or exploit banking services for unlawful profit. 
It undermines the integrity of financial institutions, leads to significant financial losses, and erodes trust among customers and stakeholders.

This simulator aims to generate banking activities including transactions following known fraud patterns. Even if deviating from these known patterns, new emerging frauds should be detected as anomalies in comparison of the activites observed in normal bevahiour, or at minimal be scored with an higher risk. In a nutshell there is a normal business, frauds following known patterns and the unknown.
    
## Features
* real_id: id of the acting agent
* timestamp: instant of the activity in ms
* activity_type: nature of the activity
* amount: amount for the transactions, or no value for other activities
* balance: balance of the account for transactions, or no value for other activities
* merchant: Transactions are associated with realistic merchants like Amazon, Walmart, etc.
* location: Each transaction is linked to a geographical location.
* device: id of the device performing the activity
* network: 	id of the network performing the activity
* compromised_device: is the device hacked
* compromised_network: is the network hacked
* fraud: the label identicating if the activity is part of a fraud or not

## How to Run
1. To avoid project conflicts, create a virtual environment `fraud_env`(necessary only the first time you run the code):
    ```
    python -m venv fraud_env
    ```
2. Activate the environment (always):
    ```
    source fraud_env/bin/activate
    ```
2. Install the required Python dependencies (necessary only the first time you run the code):
    ```
    pip install -r requirements.txt
    ```
3. Run the simulator by executing the `simulator.py` script wiht the specified inputs (number of activities, number of agents (fraudulent+legitimate), proportion of fraudulent agents):
    ```
    python src/simulator.py --nb_activities 1000 --nb_agents 10 --pr_fraudulent 0.3
    ```

## Data
The generated dataset contains x activities (both legitimate and fraudulent) with associated geographical and device features:
- [100 activities](data/fraud_simulation_100_activities.csv)
- [1K activities](data/fraud_simulation_1K_activities.csv)
- [10K activities](data/fraud_simulation_10K_activities.csv)
- [100K activities](data/fraud_simulation_100K_activities.csv)
- [1M activities](data/fraud_simulation_1M_activities.csv)

## Links
[PaySim simulator project](https://github.com/EdgarLopezPhD/PaySim)

