from datetime import datetime, timedelta
import random
import re
import pytz
from dateutil.parser import isoparse
from utilities import format_timestamp, generate_random_hash

def parse_velocity(velocity):
    """
    Parses a velocity string and returns a timedelta representing the minimum gap between transactions.
    Ensures that all extracted numbers are valid.
    """
    velocity = velocity.lower()
    print("Parsing velocity:", velocity)

    # Handle specific cases
    if "hour" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*within\s*(\d+)\s*hour", velocity)
        if match:
            min_t, max_t, hours = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t  # If max_t is None, use min_t
            hours = int(hours) if hours else 1      # Default to 1 hour if missing
            interval = timedelta(hours=hours / max_t)
            return interval

    elif "day" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*day", velocity)
        if match:
            min_t, max_t, days = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t
            days = int(days) if days else 1  # Default to 1 day if missing
            interval = timedelta(days=days / max_t)
            return interval

    elif "week" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*week", velocity)
        if match:
            min_t, max_t, weeks = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t
            weeks = int(weeks) if weeks else 1  # Default to 1 week if missing
            interval = timedelta(days=(7 * weeks) / max_t)
            return interval

    elif "minute" in velocity:
        interval = timedelta(minutes=random.randint(1, 10))
        return interval

    # Default fallback
    return timedelta(minutes=random.randint(5, 30))


#  Generate Transaction Amount
def generate_amount_from_strategy(strategy):
    """Generates a realistic transaction amount based on the strategy-defined range."""
    amount_range = strategy.get("typical_amount_range", "$10 - $500")
    
    # Extract min and max amount from the strategy range
    min_amount, max_amount = [
        float(x.replace("$", "").replace(",", "")) for x in amount_range.split(" - ")
    ]
    
    return round(random.uniform(min_amount, max_amount), 2)


#  Assign Initial Balance
def assign_initial_balance(amount=None):
    """Assigns an initial balance that is always greater than the amount if provided."""
    min_balance = 1000  # Minimum starting balance
    max_balance = 100000  # Maximum starting balance
    balance = random.randint(min_balance, max_balance)
    
    if amount and balance <= amount:
        balance += random.randint(int(amount), int(amount) * 2)  # Ensure balance > amount
    
    return balance


#  Select a Valid Location Based on the Strategy
REGION_TO_CITIES = {
    "Domestic US": ["New York, USA", "Los Angeles, USA", "Chicago, USA"],
    "Europe": ["Paris, France", "Berlin, Germany", "Madrid, Spain", "Rome, Italy"],
    "East Asia": ["Shanghai, China", "Tokyo, Japan", "Seoul, South Korea"],
    "South America": ["São Paulo, Brazil", "Buenos Aires, Argentina"],
    "Middle East": ["Dubai, UAE", "Riyadh, Saudi Arabia"]
}

def select_valid_location(geographic_focus):
    """Selects a valid city based on the strategy's geographic focus."""
    possible_cities = []
    
    for region in geographic_focus:
        if region in REGION_TO_CITIES:
            possible_cities.extend(REGION_TO_CITIES[region])
    
    if not possible_cities:
        print("No valid cities found in the strategy's geographic focus. Choosing randomly.\n")
        possible_cities = sum(REGION_TO_CITIES.values(), [])  # Flatten list of cities
    
    print(f"Possible cities for {geographic_focus}: {possible_cities}")
    chosen_city = random.choice(possible_cities)
    print(f"Selected location: {chosen_city}\n")
    return chosen_city


#  Generate a Realistic IP Address
LOCATION_TO_IP_RANGES = {
    "New York, USA": "73.56",
    "Los Angeles, USA": "172.58",
    "Chicago, USA": "98.23",
    "Paris, France": "51.75",
    "Berlin, Germany": "88.99",
    "Madrid, Spain": "213.97",
    "Rome, Italy": "151.13",
    "Shanghai, China": "203.195",
    "Tokyo, Japan": "210.153",
    "Seoul, South Korea": "121.254",
    "São Paulo, Brazil": "200.98",
    "Buenos Aires, Argentina": "190.3",
    "Dubai, UAE": "94.200",
    "Riyadh, Saudi Arabia": "188.48"
}

def generate_realistic_ip(location):
    """Generates an IP address consistent with the given location."""
    base_ip = LOCATION_TO_IP_RANGES.get(location, f"{random.randint(1, 255)}.{random.randint(1, 255)}")
    return f"{base_ip}.{random.randint(1, 255)}.{random.randint(1, 255)}"


#  Generate Timestamps
TIMEZONE_MAPPING = {
    "New York, USA": "America/New_York",
    "Los Angeles, USA": "America/Los_Angeles",
    "Chicago, USA": "America/Chicago",
    "Paris, France": "Europe/Paris",
    "Berlin, Germany": "Europe/Berlin",
    "Madrid, Spain": "Europe/Madrid",
    "Rome, Italy": "Europe/Rome",
    "Shanghai, China": "Asia/Shanghai",
    "Tokyo, Japan": "Asia/Tokyo",
    "Seoul, South Korea": "Asia/Seoul",
    "São Paulo, Brazil": "America/Sao_Paulo",
    "Buenos Aires, Argentina": "America/Argentina/Buenos_Aires",
    "Dubai, UAE": "Asia/Dubai",
    "Riyadh, Saudi Arabia": "Asia/Riyadh"
}

def generate_local_and_bank_timestamp(location, global_clock, last_tx, strategy):
    """
    Generates a local timestamp first, then maps it to UTC (bank timestamp) while enforcing order.
    """
    local_tz = pytz.timezone(TIMEZONE_MAPPING.get(location, "UTC"))
    # Handle first transaction case
    # Handle first transaction case (if there's no last_tx)
    if last_tx is None:
        print(f"No previous transaction on that account. Using global clock with random delta.")
        # Parse the global_clock (assumed to be in UTC)
        global_clock_dt = isoparse(global_clock)
        # Add a random delta between 1 and 10 minutes (adjust as needed)
        random_delta = timedelta(minutes=random.randint(1, 10))
        bank_time = global_clock_dt + random_delta
        # Convert the new UTC time to local time
        local_time = bank_time.astimezone(local_tz)
    else:
        last_time = isoparse(last_tx["local_timestamp"])
        min_gap = parse_velocity(strategy.get("velocity", "1-2 transactions per day"))
        print("Gap between transactions:", min_gap)
        local_time = last_time + min_gap
        print(f"Last transaction at {last_time}. New transaction at {local_time}")
        # Convert local time to UTC
        bank_time = local_time.astimezone(pytz.utc)

    return format_timestamp(local_time.isoformat()), format_timestamp(bank_time.isoformat())


#  Generate Static Activity
def generate_static_activity(strategy, user_id, account_id, global_clock, last_tx):
    """
    Generates a legitimate or initial fraudulent activity.
    """
    tx_type = random.choice(strategy.get("transaction_types_involved", ["Purchase"]))
    tx_amount = generate_amount_from_strategy(strategy)
    tx_location = select_valid_location(strategy.get("geographic_focus", ["Domestic US"]))
    tx_ip = generate_realistic_ip(tx_location)
    local_timestamp, bank_timestamp = generate_local_and_bank_timestamp(tx_location, global_clock, last_tx, strategy)
    # Check for fields that can be null
    if any(word in tx_type.lower() for word in ["transfer out"]):
        recipient_ids = strategy.get("common_recipient_ids", [])
        recipient_banks = strategy.get("common_recipient_banks", [])
        if recipient_ids:
            recipient_id = random.choice(recipient_ids)
        if recipient_banks:
            recipient_bank = random.choice(recipient_banks)
    else:
        recipient_id = None
        recipient_bank = None
    if any(word in tx_type.lower() for word in ["purchase", "sale"]):
        merchant_names = strategy.get("common_merchant_names", [])
        if merchant_names:
            merchant_name = random.choice(merchant_names)
    else:
        merchant_name = None
    
    return {
        "user_id": user_id,
        "bank_timestamp": bank_timestamp,
        "local_timestamp": local_timestamp,
        "account_id": account_id,
        "type": tx_type,
        "amount": tx_amount,
        "location": tx_location,
        "ip_address": tx_ip,
        "device_id": random.choice(strategy.get("common_devices", ["Unknown Device"])),
        "network_type": random.choice(strategy.get("network_types", ["Wi-Fi", "Cellular"])),
        "merchant_name": merchant_name,
        "recipient_id": recipient_id,
        "recipient_bank": recipient_bank
    }



def assign_activity_fields(tx, user_id, behavior_type, fraud_label):
    """
    Add necessary metadata fields.
    
    Ensures:
    - Proper fraud labeling and behavior assignment.
    """

    # Assign user and fraud metadata
    tx["user_id"] = user_id
    tx['transaction_id'] = f"TXN-{generate_random_hash(10)}"
    tx["behavior_type"] = behavior_type
    tx["fraud_label"] = fraud_label
    return tx


def enforce_timestamp_order(tx, last_tx):
    """
    Ensures that tx["bank_timestamp"] is strictly later than last_tx["bank_timestamp"],
    with a realistic minimum time gap based on location differences.
    
    If the new transaction (tx) is from a different location than last_tx,
    a minimum gap of 60 minutes is enforced; otherwise, a gap of 1 minute is used.
    
    The function also updates tx["local_timestamp"] to preserve the original 
    time difference between bank_timestamp and local_timestamp.
    """
    try:
        last_ts = isoparse(last_tx["bank_timestamp"])
        orig_bank = isoparse(tx["bank_timestamp"])
        orig_local = isoparse(tx["local_timestamp"])
    except Exception as e:
        print("Error parsing timestamps:", e)
        return tx

    # Determine minimum required gap based on location difference
    if tx.get("location") != last_tx.get("location"):
        min_gap = timedelta(minutes=60)  # Longer gap when locations differ
    else:
        min_gap = timedelta(minutes=1)   # Shorter gap when locations are the same

    # Check if the new bank_timestamp is at least last_ts + min_gap
    if orig_bank <= last_ts + min_gap:
        print(f"Timestamps not strictly increasing: {last_ts} vs. {orig_bank}")
        new_bank = last_ts + min_gap
        tx["bank_timestamp"] = new_bank.isoformat()
        # Preserve the original time delta between bank_timestamp and local_timestamp
        delta = orig_bank - orig_local
        new_local = new_bank - delta
        tx["local_timestamp"] = new_local.isoformat()
        print(f"Adjusted timestamps: new bank_timestamp set to {tx['bank_timestamp']}, new local_timestamp set to {tx['local_timestamp']}")
    return tx