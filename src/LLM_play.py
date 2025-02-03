import json
import subprocess
from IPython import embed

def generate_next_activity(agent_context):
    """
    agent_context: a dict with keys corresponding to the placeholders in the prompt.
    """
    prompt = f"""
You are simulating a bank's activity log. Based on the given agent's recent activity, generate the next transaction record. The output must be in valid JSON format with exactly the following keys:

- "real_id": integer (the agent's actual id)
- "virtual_id": integer (if applicable, else same as real_id)
- "timestamp": string in "YYYY-MM-DD HH:MM:SS" format (the time of the transaction)
- "delta_time": string representing a time duration (e.g., "0 days 00:05:00")
- "behavior": string (one of "legitimate", "money_laundering", etc.)
- "initial_balance": float (balance before the transaction)
- "activity_type": string (for example: "POS Purchase", "ATM Withdrawal", "Wire Transfer", "Close Account", etc.)
- "granted": boolean (True if the transaction was approved)
- "amount": float (the transaction amount; negative for withdrawals)
- "balance": float (the new balance after the transaction)
- "initial_country": string (the country where the account was originally opened)
- "location": string (the country or location from where the transaction is made)
- "device": string (for example: "iPhone", "Android", "Windows Laptop", etc.)
- "network": string (for example: "Public WiFi", "Home WiFi", "Corporate Network", etc.)
- "compromised_device": integer (0 or 1 indicating if the device is compromised)
- "compromised_network": integer (0 or 1 indicating if the network is compromised)
- "agent_type": string (for example: "static" or "traveler")
- "is_fraudster": integer (0 for legitimate, 1 for fraudster)

Context:
Agent Info:
- real_id: {agent_context["real_id"]}
- virtual_id: {agent_context["virtual_id"]}
- Previous Timestamp: {agent_context["previous_timestamp"]}
- Previous Activity: {agent_context["previous_activity"]}
- Initial Balance: {agent_context["initial_balance"]}
- Behavior: {agent_context["behavior"]}
- Last Transaction Amount: {agent_context["last_transaction_amount"]}
- Initial Country: {agent_context["initial_country"]}
- Location: {agent_context["location"]}
- Device: {agent_context["device"]}
- Network: {agent_context["network"]}
- Compromised Device: {agent_context["compromised_device"]}
- Compromised Network: {agent_context["compromised_network"]}
- Agent Type: {agent_context["agent_type"]}
- Is Fraudster: {agent_context["is_fraudster"]}

Predict the next transaction record ensuring that the new "timestamp" is later than the "Previous Timestamp" by the "delta_time".
Respond only with the valid JSON.
    """

    # Call your chosen model (e.g., "mistral" or "mixtral")
    command = ["ollama", "run", "mistral", prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Model output was:")
        print(output)
        return None


agent_context = {
    "real_id": 2,
    "virtual_id": 2,
    "previous_timestamp": "2025-01-31 03:03:53",
    "previous_activity": "Purchase Luxury Goods",
    "initial_balance": 6384.4717,
    "behavior": "money_laundering",
    "last_transaction_amount": -3995.9016,
    "initial_country": "China",
    "location": "Canada",
    "device": "iPhone",
    "network": "Home WiFi",
    "compromised_device": 0,
    "compromised_network": 0,
    "agent_type": "traveler",
    "is_fraudster": 1
}

next_activity = generate_next_activity(agent_context)
embed()
print(next_activity)
