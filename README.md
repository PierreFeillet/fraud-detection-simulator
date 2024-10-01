
# Fraud Detection Simulator

## Overview
This repository provides a Python-based simulation for fraud detection in a banking system. The simulator generates both legitimate and fraudulent transactions, as well as suspicious account activities such as password changes and failed login attempts.

## Features
- **Behavioral Catalog**: Fraud behaviors are defined in a catalog, making it easy to configure different types of fraud scenarios.
- **Fraudulent Activities**: Simulates a range of fraudulent activities including failed login attempts, password changes, suspicious withdrawals, and more.

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
- A sample dataset of 100 activities is included in the `data` folder. This dataset contains both legitimate and fraudulent activities along with the corresponding risk levels and fraud labels.

## Customization
The fraud scenarios can be customized through the `behavioral_catalog` in the `simulator.py` script. You can add new fraud types, modify existing behaviors, and adjust the probabilities for different activities.
