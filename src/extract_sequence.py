# Behavioral catalog for various fraudulentulent behaviors with time series and activity sequences
import numpy as np
import random
# Locations
locations = ["New York, USA", "Los Angeles, USA", "London, UK", "Tokyo, Japan", "Paris, France", "Berlin, Germany"]
# Define location weights (if you want some locations to appear more frequently)
location_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Adjust these weights as needed
LOCATIONS = {
    "USA": [
        ("New York", 0.2),
        ("Los Angeles", 0.2)
    ],
    "UK": [
        ("London", 0.15)
    ],
    "France": [
        ("Paris", 0.1)
    ],
    "Italy": [
        ("Rome", 0.1)
    ],
    "Germany":[
        ("Berlin", 0.1)
    ],
    "Japan":[
        ("Tokyo", 0.15)
    ]
}

# Merchants
# Define allowed transaction activity_types for each merchant
TRANSACTION_activity_type_MERCHANTS = {
    "purchase": [
        ("Walmart", 0.15),
        ("Best Buy", 0.2),
        ("Target", 0.1),
        ("Starbucks", 0.1),
        ("Apple", 0.1),
        ("Amazon", 0.25),
        ("PayPal", 0.1)
    ],
    "deposit": [
        ("Bank", 0.4),
        ("ATM", 0.3),
        ("PayPal", 0.3)
    ],
    "withdrawal": [
        ("Bank", 0.5),
        ("ATM", 0.5)
    ]
}

# Devices and device weights
devices = ["iPhone", "Android", "Windows Laptop", "MacBook", "Linux PC", "iPad"]
device_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]  # Weights representing how frequently each device is used

# Networks and network weights
networks = ["Home WiFi", "Public WiFi", "Mobile Network", "Corporate Network"]
network_weights = [0.4, 0.3, 0.2, 0.1]  # Weights indicating the frequency of each network

# List of transactions. If not in this list a certain activity is considered an activity
possible_transactions = ["withdrawal", "deposit", "purchase"]

# Behavioral catalog for different fraudulent activity_types

