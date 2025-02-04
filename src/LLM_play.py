import ollama
from datetime import datetime, timedelta
import json
from IPython import embed

# Sample agent data
agent_type = "traveler"
initial_country = "USA"
behavior = "frequent traveler"
is_fraudster = 0
previous_activity = "purchase"
previous_timestamp = datetime.now() - timedelta(hours=1, minutes=30)  # 1.5 hours ago
previous_location = "France"
previous_amount = 120.50  # USD

# Construct the LLM prompt
prompt = (
    f"You are a structured data generator. You must always return a response in strict JSON format, without any additional text. "
    f"Given the following agent details, predict their next financial activity:\n\n"
    f"Agent Type: {agent_type} (Options: 'traveler', 'static')\n"
    f"Residence Country: {initial_country}\n"
    f"Behavior Type: {behavior}\n"
    f"Is Fraudster: {is_fraudster} (0 = Legitimate, 1 = Fraudster)\n"
    f"Previous Activity: {previous_activity} (Options: 'withdrawal', 'deposit', 'purchase', 'login', 'account takeover', etc.)\n"
    f"Previous Timestamp: {previous_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"Previous Location: {previous_location}\n"
    f"Previous Amount: {previous_amount} (Use 0 for non-transaction activities)\n\n"
    f"Constraints:\n"
    f"- The predicted activity must align with the agent's **behavior**.\n"
    f"- Timestamp must be after the previous timestamp, with a delay between **5 minutes and 2 hours**.\n"
    f"- The activity must be consistent with past behavior and agent type.\n"
    f"- If the activity is a **transaction**, predict a reasonable amount based on behavior (e.g., small purchases <$100, large withdrawals >$500).\n"
    f"- If the agent is a **fraudster**, they are more likely to conduct suspicious activities such as high-value transactions, international transfers, or account takeovers.\n"
    f"- Return **only** JSON format, like this:\n"
    f"  {{\"activity\": \"...\", \"timestamp\": \"YYYY-MM-DD HH:MM:SS\", \"location\": \"...\", \"amount\": ...}}\n"
)


# Send request to Ollama
response = ollama.chat(model="deepseek-r1", messages=[{"role": "system", "content": "You are an AI trained to return structured JSON outputs. Always return valid JSON, nothing else."},{"role": "user", "content": prompt}])
embed()
# Parse the response
try:
    data = json.loads(response['message']['content'])
    print("LLM Response:", data)
except Exception as e:
    print("Error parsing response:", str(e))
