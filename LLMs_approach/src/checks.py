import pandas as pd

import pandas as pd
import numpy as np

def validate_balance(csv_path):
    """
    Validates the balance computation in a CSV file containing activity data.
    
    Args:
        csv_path (str): Path to the CSV file containing activity data.
        
    Returns:
        pd.DataFrame: A DataFrame containing rows where the balance computation is incorrect.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure numerical columns are of correct activity_types
    numeric_columns = ["initial_balance", "amount", "balance"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Compute the expected balance
    def compute_expected_balance(row):
        if row["granted"]:
            if row["action"] in ["withdrawal", "purchase"]:
                return row["initial_balance"] - row["amount"]
            elif row["action"] == "deposit":
                return row["initial_balance"] + row["amount"]
        return row["initial_balance"]  # If not granted, balance should remain the same

    # Calculate the expected balance
    df["expected_balance"] = df.apply(compute_expected_balance, axis=1)

    # Use np.isclose for floating-point comparisons to handle precision issues
    invalid_rows = df[~np.isclose(df["balance"], df["expected_balance"], atol=1e-6)]
    
    return invalid_rows

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "/home/molocco/fraud-detection-simulator/data/fraud_simulation_1K_activities.csv"

    # Validate the balance
    invalid_rows = validate_balance(csv_path)

    # Check the output
    if not invalid_rows.empty:
        print("Invalid balance computations found:")
        print(invalid_rows)
    else:
        print("All balance computations are correct.")