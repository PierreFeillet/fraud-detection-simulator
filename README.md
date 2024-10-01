
# Fraud Detection Simulator

## Overview
This project simulates both legitimate and fraudulent transactions in a banking system in a multi-agent architecture. The simulation includes geographical information, merchants, devices, networks, and analyzes whether devices or networks are compromised.

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
- The generated dataset contains 100 activities (both legitimate and fraudulent) with associated geographical and device features.
