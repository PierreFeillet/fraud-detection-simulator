import ollama
import time
import json
import pandas as pd
import os
import uuid
import re

# Define available models
models = {
    "Mistral": "mistral",
    "DeepSeek-R1": "deepseek-r1"
}
'''
"Phi-2": "phi",
"Gemma2:2b": "gemma2"\
"Llama"
'''

# Define test parameters
user_id = "AI-12345"
global_clock = '2025-03-01 09:00:00'
fraud_label = 1
n_trials = 10
csv_filename = "benchmark_results.csv"

# Function to extract JSON from response
def extract_json(text):
    retry_count = 0
    while retry_count < 3:  
        matches = re.findall(r'```json\s*(\[\s*{.*?}\s*\])\s*```', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)
        if not matches:
            retry_count += 1
            return 'retry', retry_count

        combined_activities = []
        for json_text in matches:
            json_text_cleaned = re.sub(r'//.*', '', json_text)
            try:
                data = json.loads(json_text_cleaned)
                combined_activities.extend(data if isinstance(data, list) else [data])
                return combined_activities, retry_count
            except json.JSONDecodeError:
                retry_count += 1

    return 'retry', retry_count

# Load existing benchmark results to avoid duplications
if os.path.exists(csv_filename):
    df_existing = pd.read_csv(csv_filename)
    completed_tests = set(zip(df_existing["Model"], df_existing["Profile type"], df_existing["Trial"]))
else:
    df_existing = pd.DataFrame()
    completed_tests = set()

# Load fraudulent strategies
def load_existing_strategies(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

fraudulent_strategies = load_existing_strategies("strategies/fraud_strategies.json")
behaviors = list(fraudulent_strategies.keys())

# Run benchmark
for model_name, model_id in models.items():
    for behavior in behaviors:
        strategy = fraudulent_strategies[behavior]
        for trial in range(1, n_trials + 1):

            # Check if the test was already completed
            if (model_name, behavior, trial) in completed_tests:
                print(f"Skipping {model_name} - {behavior} (Trial {trial}), already completed.")
                continue

            print(f"Testing {model_name} - {behavior} (Trial {trial}/{n_trials})...")

            # Generate prompt
            prompt = f"Simulate transaction behavior for {behavior}. User ID: {user_id}."

            # Measure response time
            start_time = time.time()
            response = ollama.chat(model=model_id, messages=[{"role": "user", "content": prompt}])
            end_time = time.time()
            response_time = round(end_time - start_time, 2)

            # Extract JSON
            response_content = response['message']['content']
            extracted_json, retry_count = extract_json(response_content)

            # Store result
            row_data = {
                "Model": model_name,
                "Profile type": behavior,
                "Trial": trial,
                "Response Time (s)": response_time,
                "Valid JSON": extracted_json != "retry",
                "Retries": retry_count,
                "Raw Response": response_content,
                "Generated Transactions": extracted_json if extracted_json != "retry" else "Invalid JSON output"
            }

            # Append row to CSV file immediately
            pd.DataFrame([row_data]).to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)

            print(f"Saved result for {model_name} - {behavior} (Trial {trial})")

print("Benchmark completed and saved.")

# Read df_results
df_results = pd.read_csv('benchmark_results.csv')
print(df_results)
# Compute statistics
stats_df = df_results.groupby(["Model", "Profile Type"]).agg({
    "Valid JSON": ["count", "sum", lambda x: 100 * (1 - x.mean())],  # Total, valid count, failure rate (%)
    "Retries": ["mean", "max"],  # Average & max retries
    "Response Time (s)": ["mean", "min", "max"]  # Response time stats
}).reset_index()

import pandas as pd
import matplotlib.pyplot as plt

# Read the benchmark results from CSV
df_results = pd.read_csv('../benchmark_results.csv')

# Compute statistics
stats_df = df_results.groupby("Model").agg({
    "Valid JSON": ["count", "sum", lambda x: 100 * (1 - x.mean())],  # Total, valid count, failure rate (%)
    "Retries": ["mean", "max"],  # Average & max retries
    "Response Time (s)": ["mean", "min", "max"]  # Response time stats
}).reset_index()

# Rename columns for clarity
stats_df.columns = [
    "Model", "Profile Type", "Total Trials", "Valid JSON Count", "Failure Rate (%)",
    "Avg Retries", "Max Retries", "Avg Response Time (s)", "Min Response Time (s)", "Max Response Time (s)"
]

print(stats_df)
# Create a single figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot 1: Failure Rate Comparison
axes[0].bar(stats_df["Model"], stats_df["Failure Rate (%)"], alpha=0.75)
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Failure Rate (%)")
axes[0].set_title("Failure Rate Comparison Across Models")
axes[0].set_ylim(0, 100)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: Response Time Distribution
axes[1].bar(stats_df["Model"], stats_df["Avg Response Time (s)"], label="Avg", color="orange", alpha=0.65)
axes[1].scatter(stats_df["Model"], stats_df["Min Response Time (s)"], color="green", label="Min", marker="o")
axes[1].scatter(stats_df["Model"], stats_df["Max Response Time (s)"], color="red", label="Max", marker="o")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Response Time (s)")
axes[1].set_title("Response Time Distribution Across Models")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 3: Average Retries per Model
axes[2].bar(stats_df["Model"], stats_df["Avg Retries"], color="orange", alpha=0.65)
axes[2].set_xlabel("Model")
axes[2].set_ylabel("Average Retries")
axes[2].set_title("Average Number of Retries per Model")
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("benchmark_analysis.png")

# Display the figure
plt.show()


