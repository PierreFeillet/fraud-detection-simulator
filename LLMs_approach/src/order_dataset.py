import os
import pandas as pd

# Your file path
file_path = "data/mistral-large/fraud_simulation_activities_1K.csv"

# 1. Parse the last part between '_' and '.csv' to capture "1K"
base_name = os.path.basename(file_path)          # e.g. "fraud_simulation_activities_1K.csv"
name_part, ext = os.path.splitext(base_name)     # name_part = "fraud_simulation_activities_1K", ext = ".csv"
split_parts = name_part.split("_")               # ["fraud", "simulation", "activities", "1K"]
last_part = split_parts[-1]                      # "1K"

# 2. Build the sorted filename using the last part
sorted_filename = f"sorted_transactions_{last_part}.csv"
output_path = os.path.join(os.path.dirname(file_path), sorted_filename)

# 3. Read and sort the CSV file by 'bank_timestamp'
df = pd.read_csv(file_path)
df['bank_timestamp'] = pd.to_datetime(df['bank_timestamp'])
df_sorted = df.sort_values(by='bank_timestamp')

# 4. Save the sorted data to a new CSV file
df_sorted.to_csv(output_path, index=False)

print(f"Sorted dataset saved to {output_path}")
