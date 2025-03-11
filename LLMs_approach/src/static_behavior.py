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

    # Handle specific cases
    if "hour" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*within\s*(\d+)\s*hour", velocity)
        if match:
            min_t, max_t, hours = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t  # If max_t is None, use min_t
            hours = int(hours) if hours else 1      # Default to 1 hour if missing
            return timedelta(hours=hours / max_t)

    elif "day" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*day", velocity)
        if match:
            min_t, max_t, days = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t
            days = int(days) if days else 1  # Default to 1 day if missing
            return timedelta(days=days / max_t)

    elif "week" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*week", velocity)
        if match:
            min_t, max_t, weeks = match.groups()
            min_t = int(min_t)
            max_t = int(max_t) if max_t else min_t
            weeks = int(weeks) if weeks else 1  # Default to 1 week if missing
            return timedelta(days=(7 * weeks) / max_t)

    elif "minute" in velocity:
        return timedelta(minutes=random.randint(1, 10))

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
    min_balance = 100  # Minimum starting balance
    max_balance = 10000  # Maximum starting balance
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
        possible_cities = sum(REGION_TO_CITIES.values(), [])  # Flatten list of cities

    return random.choice(possible_cities)


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

def generate_local_and_bank_timestamp(location, last_tx, strategy):
    """
    Generates a local timestamp first, then maps it to UTC (bank timestamp) while enforcing order.
    """
    local_tz = pytz.timezone(TIMEZONE_MAPPING.get(location, "UTC"))
    print(last_tx)
    # Handle first transaction case
    # Handle first transaction case (if there's no last_tx)
    if last_tx is None or "local_timestamp" not in last_tx:
        print(f"No previous transaction found. Initializing timestamp for {location}")
        local_time = datetime.now(local_tz)
    else:
        last_time = isoparse(last_tx["local_timestamp"])
        min_gap = parse_velocity(strategy.get("velocity", "1-2 transactions per day"))
        local_time = last_time + min_gap

    # Convert local time to UTC
    bank_time = local_time.astimezone(pytz.utc)

    return format_timestamp(local_time.isoformat()), format_timestamp(bank_time.isoformat())


#  Generate Static Activity
def generate_static_activity(strategy, user_id, account_id, current_balance, last_timestamp):
    """
    Generates a legitimate or initial fraudulent activity.
    """
    tx_type = random.choice(strategy.get("transaction_types_involved", ["Purchase"]))
    tx_amount = generate_amount_from_strategy(strategy)
    tx_location = select_valid_location(strategy.get("geographic_focus", ["Domestic US"]))
    tx_ip = generate_realistic_ip(tx_location)
    tx_device = random.choice(strategy.get("common_devices", ["Unknown Device"]))
    local_timestamp, bank_timestamp = generate_local_and_bank_timestamp(tx_location, last_timestamp, strategy)

    return {
        "user_id": user_id,
        "bank_timestamp": bank_timestamp,
        "local_timestamp": local_timestamp,
        "account_id": account_id,
        "type": tx_type,
        "amount": tx_amount,
        "location": tx_location,
        "ip_address": tx_ip,
        "device_id": tx_device,
        "network_type": random.choice(strategy.get("network_types", ["Wi-Fi", "Cellular"])),
    }



def assign_activity_fields(tx, user_id, behavior_type, fraud_label):
    """
    Enriches an activity transaction with timestamps, location-based details,
    and necessary metadata fields.
    
    Ensures:
    - Correct timestamp handling (bank_timestamp first, then local_timestamp).
    - Completeness of all fields (avoids missing values).
    - Proper fraud labeling and behavior assignment.
    """

    # Assign user and fraud metadata
    tx["user_id"] = user_id
    tx['transaction_id'] = f"TXN-{generate_random_hash(10)}"
    tx["behavior_type"] = behavior_type
    tx["fraud_label"] = fraud_label

    # Assign missing but required fields
    tx.setdefault("device_id", "Unknown Device")
    tx.setdefault("network_type", "Unknown Network")
    tx.setdefault("merchant_name", None)
    tx.setdefault("recipient_id", None)
    tx.setdefault("recipient_bank", None)

    # Assign timestamps (Bank Timestamp first, then compute Local Timestamp)
    location = tx.get("location", "Unknown Location")
    last_tx = tx.get("last_transaction", None)  # If available, pass previous tx

    # Generate timestamps with strict order enforcement
    bank_timestamp, local_timestamp = generate_local_and_bank_timestamp(location, last_tx, strategy={})  # Pass strategy if needed
    tx["bank_timestamp"] = bank_timestamp
    tx["local_timestamp"] = local_timestamp

    return tx
