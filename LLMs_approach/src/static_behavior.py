

from datetime import datetime, timedelta
import random
from pytz import timezone
from dateutil.parser import isoparse  # Install python-dateutil if needed
import re
from datetime import datetime, timedelta
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dateutil.parser import isoparse

def parse_velocity(velocity):
    """
    Parses a velocity string and returns a timedelta representing the minimum gap between transactions.
    """
    velocity = velocity.lower()

    # Handle specific cases
    if "hour" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*within\s*(\d+)\s*hour", velocity)
        if match:
            min_t, max_t, hours = match.groups()
            min_t, max_t = int(min_t), int(max_t) if max_t else int(min_t)
            return timedelta(hours=int(hours) / max_t)

    elif "day" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*day", velocity)
        if match:
            min_t, max_t, days = match.groups()
            min_t, max_t = int(min_t), int(max_t) if max_t else int(min_t)
            return timedelta(days=int(days) / max_t)

    elif "week" in velocity:
        match = re.search(r"(\d+)-?(\d+)?\s*transactions\s*per\s*(\d+)?\s*week", velocity)
        if match:
            min_t, max_t, weeks = match.groups()
            min_t, max_t = int(min_t), int(max_t) if max_t else int(min_t)
            return timedelta(days=(7 * int(weeks)) / max_t)

    elif "minute" in velocity:
        return timedelta(minutes=random.randint(1, 10))

    # Default fallback
    return timedelta(minutes=random.randint(5, 30))


def generate_amount_from_strategy(strategy):
    """Generates a realistic transaction amount based on the strategy-defined range."""
    amount_range = strategy.get("typical_amount_range", "$10 - $500")
    
    # Extract min and max amount from the strategy range
    min_amount, max_amount = [
        float(x.replace("$", "").replace(",", "")) for x in amount_range.split(" - ")
    ]
    
    # Generate a random transaction amount within range
    return round(random.uniform(min_amount, max_amount), 2)

def assign_initial_balance(amount=None):
    """Assigns an initial balance that is always > amount if provided."""
    min_balance = 100  # Minimum starting balance
    max_balance = 10000  # Maximum starting balance
    balance = random.randint(min_balance, max_balance)
    
    if amount and balance <= amount:
        balance += random.randint(int(amount), int(amount) * 2)  # Ensure balance > amount
    
    return balance

# Mapping of regions to their major cities
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

    # Match the focus regions to mapped cities
    for region in geographic_focus:
        if region in REGION_TO_CITIES:
            possible_cities.extend(REGION_TO_CITIES[region])

    # If no valid region found, fall back to a random city
    if not possible_cities:
        possible_cities = sum(REGION_TO_CITIES.values(), [])  # Flatten list of cities

    return random.choice(possible_cities)

# Mapping of locations to plausible IP address ranges
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
    if location in LOCATION_TO_IP_RANGES:
        base_ip = LOCATION_TO_IP_RANGES[location]
    else:
        base_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}"

    return f"{base_ip}.{random.randint(1, 255)}.{random.randint(1, 255)}"


# Initialize external libraries
geolocator = Nominatim(user_agent="geoapiExercises")
tf = TimezoneFinder()

def generate_local_and_bank_timestamp(location, last_tx, strategy):
    """
    Generates a UTC (bank) timestamp first, then derives the local timestamp based on the transaction's location.
    
    Ensures:
    - Chronological order (timestamps must strictly increase).
    - Local timestamp follows the correct timezone offset.
    """

    # Handle first transaction case
    if not last_tx:
        bank_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    else:
        last_time = isoparse(last_tx["bank_timestamp"])

        # Determine the time interval based on the strategy’s velocity
        velocity = strategy.get("velocity", "1-2 transactions per day")
        min_gap = parse_velocity(velocity)  # Function to get timedelta based on velocity
        bank_time = last_time + min_gap

    try:
        # Fetch latitude and longitude for the location
        loc = geolocator.geocode(location)
        if not loc:
            raise ValueError(f"Location not found: {location}")

        lat, lon = loc.latitude, loc.longitude

        # Determine timezone
        tz_name = tf.timezone_at(lng=lon, lat=lat)
        if not tz_name:
            raise ValueError(f"Timezone not found for location: {location}")

        # Convert bank_time (UTC) to local_time in the determined timezone
        local_tz = pytz.timezone(tz_name)
        local_time = bank_time.astimezone(local_tz)

        return bank_time.isoformat(), local_time.isoformat()

    except Exception as e:
        print(f"Warning: Could not determine timezone for {location}. Using UTC. Error: {e}")
        return bank_time.isoformat(), bank_time.isoformat()  # Fallback to UTC if timezone lookup fails

""""

# Mapping of locations to timezones
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
    
    # Get the timezone of the selected location
    local_tz = timezone(TIMEZONE_MAPPING.get(location, "UTC"))
    
    # Handle first transaction case
    if not last_tx:
        local_time = datetime.now(local_tz)
    else:
        last_time = isoparse(last_tx["local_timestamp"])

        # Determine the time interval based on the strategy’s velocity
        velocity = strategy.get("velocity", "1-2 transactions per day")
        min_gap = parse_velocity(velocity)
        local_time = last_time + min_gap

    # Convert local time to bank timestamp (UTC)
    bank_time = local_time.astimezone(timezone.utc)

    return local_time.isoformat(), bank_time.isoformat()
"""

def generate_static_activity(strategy, user_id, account_id, current_balance, last_timestamp):
    """
    Generates a legitimate activity for a user based on the given strategy.
    This function is also used to initialize the first fraudulent activity in a sequence.
    """
    # 1. Choose the transaction type from the strategy
    tx_type = random.choice(strategy.get("transaction_types_involved", ["Purchase"]))

    # 2. Generate a transaction amount based on the strategy
    tx_amount = generate_amount_from_strategy(strategy)

    # 3. Select a location based on the strategy's geographic focus
    tx_location = select_valid_location(strategy.get("geographic_focus", ["Domestic US"]))

    # 4. Generate an IP address consistent with the location
    tx_ip = generate_realistic_ip(tx_location)

    # 5. Choose a device from the strategy
    tx_device = random.choice(strategy.get("common_devices", ["Unknown Device"]))

    # 6. Generate a timestamp following the strategy’s time pattern
    local_timestamp, bank_timestamp = generate_local_and_bank_timestamp(tx_location, last_timestamp, strategy)
    
    # 7. Assign merchant, recipient, and bank details
    if tx_type == "Purchase":
        tx_merchant = random.choice(strategy.get("common_merchant_names", ["Generic Store"]))
        tx_recipient_id, tx_recipient_bank = None, None
    elif tx_type in ["Transfer Out", "Withdrawal"]:
        tx_merchant = None
        tx_recipient_id = random.choice(strategy.get("common_recipient_ids", ["REC-12345678"]))
        tx_recipient_bank = random.choice(strategy.get("common_recipient_banks", ["Bank of America"]))
    else:
        tx_merchant, tx_recipient_id, tx_recipient_bank = None, None, None
    
    # Assign network type
    tx_network_type= random.choice(strategy.get("network_types", ["Wi-Fi", "Cellular"])),

    # 8. Construct the transaction
    tx = {
        "user_id": user_id,
        "bank_timestamp": bank_timestamp,
        "local_timestamp": local_timestamp,
        "account_id": account_id,
        "type": tx_type,
        "amount": tx_amount,
        "location": tx_location,
        "ip_address": tx_ip,
        "device_id": tx_device,
        "network_type": tx_network_type,
        "merchant_name": tx_merchant,
        "recipient_id": tx_recipient_id,
        "recipient_bank": tx_recipient_bank
    }

    return tx

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
