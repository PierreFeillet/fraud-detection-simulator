from datetime import timedelta, timezone
import secrets
from dateutil.parser import isoparse  # Install python-dateutil if needed
import random

def generate_random_hash(length=8):
    """Generates a random hexadecimal string of the given length."""
    return secrets.token_hex(length // 2)

def format_timestamp(time):
    parsed_time = isoparse(time)
    st_time = parsed_time.isoformat(timespec='seconds') 
    return st_time


#  Assign Initial Balance
def assign_initial_balance(strategy):
    """Assigns an initial balance based on the strategy-defined range."""
    amount_range = strategy.get("typical_amount_range", "$1000 - $50000")
    # Extract min and max amount from the strategy range
    min_amount, max_amount = [
        float(x.replace("$", "").replace(",", "")) for x in amount_range.split(" - ")
    ]
    
    return round(random.uniform(min_amount*10, max_amount*10), 2)
# Assuming in case of transfer, the receipient is not a bank's client. ONly the account id is client

def update_balance(tx):
    """
    Computes the new balance based on the activity type and amount.
    For a deposit, adds the amount.
    For a withdrawal, transfer, purchase, or sale, subtracts the amount.
    """
    tx_type = tx.get("type", "").lower()
    amount = tx.get("amount", 0)
    #tx["granted"] = True  # Default to True if missing
    current_balance = tx["balance_before"]
    if any(word in tx_type for word in ["deposit", "contribution", "transfer in", "sale"]):
        return round(current_balance + amount,2)
    elif any(word in tx_type for word in ["withdrawal", "transfer out", "purchase"]):
        new_balance = round(current_balance - amount,2)
        if new_balance<0:
            #tx["granted"] = False
            return current_balance
        else:
            #tx["granted"] = True
            return new_balance
    else:
        return current_balance