
# Fraud Detection Simulator

## Overview
This project simulates both legitimate and fraudulent transactions in a banking system in a multi-agent architecture. The simulation includes geographical information, merchants, devices, networks, and analyzes whether devices or networks are compromised.

## Fraudulent behaviours
Behaviours are captured in the catalog file and represent several known patterns of normal or fraudulent activities:
* normal
* identity_theft
* money_laundering
* phishing
* card_skimming
    
## Features
- **Geographical Information**: Each transaction is linked to a geographical location.
- **Merchants**: Transactions are associated with merchants like Amazon, Walmart, etc.
- **Devices and Networks**: Transactions include device and network information, with the possibility of compromised devices or networks.

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
- [100 activities](data/fraud_simulation_100_activities.csv):
- [1K activities](data/fraud_simulation_1K_activities.csv):
- [10K activities](data/fraud_simulation_10K_activities.csv):
- [100K activities](data/fraud_simulation_100K_activities.csv):
- [1M activities](data/fraud_simulation_1M_activities.csv):

