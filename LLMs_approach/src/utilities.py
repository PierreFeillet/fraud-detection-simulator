from datetime import timedelta
import secrets
from dateutil.parser import isoparse  # Install python-dateutil if needed

def generate_random_hash(length=8):
    """Generates a random hexadecimal string of the given length."""
    return secrets.token_hex(length // 2)

def format_timestamp(time):
    parsed_time = isoparse(time)
    st_time = parsed_time.isoformat(timespec='seconds') 
    return st_time

def enforce_timestamp_order(tx, last_timestamp_str):
    """
    Checks if tx["bank_timestamp"] is strictly later than last_timestamp_str.
    If not, adjusts tx["bank_timestamp"] to be at least 1 minute later and updates tx["local_timestamp"]
    to preserve the original time difference.
    """
    try:
        last_ts = isoparse(last_timestamp_str)
        orig_bank = isoparse(tx["bank_timestamp"])
        orig_local = isoparse(tx["local_timestamp"])
    except Exception as e:
        print("Error parsing timestamps:", e)
        return tx

    # Compute the original difference between bank_timestamp and local_timestamp
    # (this difference may include timezone offsets)
    delta = orig_bank - orig_local  # timedelta

    if orig_bank <= last_ts:
        # Enforce a minimum gap of 1 minute
        new_bank = last_ts + timedelta(minutes=1)
        tx["bank_timestamp"] = new_bank.isoformat()
        # Adjust local_timestamp to preserve the original time difference
        new_local = new_bank - delta
        tx["local_timestamp"] = new_local.isoformat()
        print(f"Adjusted timestamps: new bank_timestamp set to {tx['bank_timestamp']}, new local_timestamp set to {tx['local_timestamp']}")
    return tx