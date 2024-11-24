import random
import pandas as pd
import numpy as np

from catalog import generate_transaction_amount, transaction_type, account_activity
from catalog import behavioral_catalog
from catalog import MERCHANT_TRANSACTION_TYPES
from catalog import merchants
from catalog import merchant_weights
from catalog import locations
from catalog import location_weights
from catalog import devices
from catalog import device_weights
from catalog import networks
from catalog import network_weights

from agent import Agent

# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Fraudster(Agent):
    def commit_fraud(self, fraud_type):
        merchant = np.random.choice(merchants, p=merchant_weights)
        
        transaction_type_choice = transaction_type(fraud_type, merchant)
        amount = generate_transaction_amount(transaction_type_choice, fraud_type)
        
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = self.is_fraud()
        compromised_network = self.is_fraud()

        if transaction_type_choice == "withdrawal" or transaction_type_choice == "purchase":
            self.balance -= amount
        else:
            self.balance += amount

        return {
            "agent_id": self.agent_id,
            "timestamp": pd.Timestamp.now(),
            "type": "fraud",
            "amount": amount,
            "balance": self.balance,
            "merchant": merchant,
            "location": location,
            "device": device,
            "network": network,
            "compromised_device": compromised_device,
            "compromised_network": compromised_network,
            "fraud": 1
        }

    def account_activity(self, fraud_type):
        activity = account_activity(fraud_type)
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = self.is_fraud()
        compromised_network = self.is_fraud()

        return {
            "agent_id": self.agent_id,
            "timestamp": pd.Timestamp.now(),
            "type": activity,
            "amount": 0,
            "balance": self.balance,
            "merchant": 0,
            "location": location,
            "device": device,
            "network": network,
            "compromised_device": compromised_device,
            "compromised_network": compromised_network,
            "fraud": 1
        }