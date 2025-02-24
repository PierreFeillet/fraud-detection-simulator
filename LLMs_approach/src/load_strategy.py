def load_strategies_from_txt(filename="strategies/fraud_strategies.txt"):
    with open(filename, "r") as file:
        content = file.read()
    
    # Split the content by the separator "---"
    strategies = content.strip().split("---\n")
    parsed_strategies = []
    
    for strategy in strategies:
        if strategy.strip():
            # Extracting profile type and strategy
            profile_type = ""
            strategy_text = ""
            
            # Split the strategy into lines
            lines = strategy.strip().split("\n")
            
            # Check for 'Profile Type' in the first line
            if lines[0].startswith("Profile Type: "):
                profile_type = lines[0].replace("Profile Type: ", "").strip()
                
                # Look for the 'Strategy:' section and collect the strategy text
                try:
                    strategy_start_index = lines.index("Strategy:") + 1
                    strategy_text = "\n".join(lines[strategy_start_index:]).strip()
                except ValueError:
                    strategy_text = ""  # In case 'Strategy:' is missing

                # Append the parsed result
                parsed_strategies.append({"Profile Type": profile_type, "Strategy": strategy_text})
            else:
                # If 'Profile Type:' is missing, skip or handle differently
                continue

    return parsed_strategies

# Example usage:
fraud_strategies = load_strategies_from_txt(filename="strategies/fraud_strategies.txt")
for fs in fraud_strategies:
    print(f"Profile Type: {fs['Profile Type']}\nStrategy:\n{fs['Strategy'][:100]}...\n")

import pandas as pd 

file_csv='/Users/molocco/IBM_secondment/fraud-detection-simulator/banking_activity_log.csv'

df = pd.read_csv(file_csv)