
# Fraud Detection Simulator

## Overview
This project simulates both legitimate and fraudulent transactions in a banking system in a multi-agent architecture. The simulation includes geographical information, merchants, devices, networks, and analyzes whether devices or networks are compromised.

Nominal and fraudulent behaviours are modeled in a catalog as Marhov chains. Each chain is a time sequence of activities with an associated probability.
Each beahviour is described through a descriptor in the - [catalog](catalog.py)
All activities participating to nominal and fraudulent behavioral sequences are meshed into a unique dataset and timeline.

## Models for fraudulent and normal behaviours
Behaviours are captured in the catalog file and represent several known patterns of normal or fraudulent activities:
* normal
* identity_theft
* money_laundering
* phishing
* card_skimming

They capture a simplified sequence of activities and can be endlessly improved to better refect state of real activities and fraud evolutions.
    
## Features
* agent_id: if of the acting agent
* timestamp: instant of the activity in ms
* type: nature of the activity
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
1. Install the required Python dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Run the simulator by executing the `simulator.py` script:
    ```
    python src/simulator.py
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

