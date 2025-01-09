import random
import pandas as pd
import numpy as np

from catalog import generate_transaction_amount, get_transaction_type, account_activity, extract_merchant
from catalog import behavioral_catalog
from catalog import locations
from catalog import location_weights
from catalog import devices
from catalog import device_weights
from catalog import networks
from catalog import network_weights

from agent import Agent

# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class LegitimateCustomer(Agent):

    def generate_transaction(self):
        transaction_type = get_transaction_type("normal")
        merchant = extract_merchant(transaction_type)
        amount = generate_transaction_amount(transaction_type, "normal")
        #transac_time = 0
        #merchant = np.random.choice(merchants, p=merchant_weights)
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = self.is_compromised()
        compromised_network = self.is_compromised()
        self.before_transaction = self.balance

        if (transaction_type == "withdrawal" or transaction_type == "purchase") and self.balance > amount:
            self.balance -= amount
            self.before_transaction = self.balance
        elif transaction_type == "deposit":
            self.balance += amount

        return {
            "agent_id": self.agent_id,
            "timestamp": pd.Timestamp.now(),
            #"time": transac_time,
            "type": transaction_type,
            "before_transaction": self.before_transaction,
            "amount": amount,
            "balance": self.balance,
            "merchant": merchant,
            "location": location,
            "device": device,
            "network": network,
            "compromised_device": compromised_device,
            "compromised_network": compromised_network,
            "fraud": 0
        }

    def account_activity(self):
        activity = account_activity("normal")
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = self.is_compromised()
        compromised_network = self.is_compromised()

        return {
            "agent_id": self.agent_id,
            "timestamp": pd.Timestamp.now(),
            "type": activity,
            "before_transaction": self.before_transaction,
            "amount": 0,
            "balance": self.balance,
            "merchant": 0,
            "location": location,
            "device": device,
            "network": network,
            "compromised_device": compromised_device,
            "compromised_network": compromised_network,
            "fraud": 0
        }