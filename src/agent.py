import random
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

# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, agent_id, balance):
        self.agent_id = agent_id
        self.balance = balance
        self.type = type=np.random.choice(["bank", "merchant", "client"], p=[0.2, 0.4, 0.4])

class LegitimateCustomer(Agent):

    def generate_transaction(self):
        merchant = np.random.choice(merchants, p=merchant_weights) #ToDo condition the actor to its type
        allowed_types = MERCHANT_TRANSACTION_TYPES[merchant] #ToDo condition the transaction type to its actor type

        transaction_type_choice = transaction_type("normal", merchant)
        amount = generate_transaction_amount(transaction_type_choice, "normal")
        
        #merchant = np.random.choice(merchants, p=merchant_weights)
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = is_fraud()
        compromised_network = is_fraud()

        if (transaction_type_choice == "withdrawal" or transaction_type_choice == "purchase") and self.balance > amount:
            self.balance -= amount
        elif transaction_type_choice == "deposit":
            self.balance += amount

        return {
            "agent_id": self.agent_id,
            "type": transaction_type_choice,
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
        compromised_device = is_fraud()
        compromised_network = is_fraud()

        return {
            "agent_id": self.agent_id,
            "type": activity,
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

class Fraudster(Agent):
    def commit_fraud(self, fraud_type):
        merchant = np.random.choice(merchants, p=merchant_weights)
        
        transaction_type_choice = transaction_type(fraud_type, merchant)
        amount = generate_transaction_amount(transaction_type_choice, fraud_type)
        
        location = np.random.choice(locations, p=location_weights)
        device = np.random.choice(devices, p=device_weights)
        network = random.choice(networks)
        compromised_device = is_fraud()
        compromised_network = is_fraud()

        if transaction_type_choice == "withdrawal" or transaction_type_choice == "purchase":
            self.balance -= amount
        else:
            self.balance += amount

        return {
            "agent_id": self.agent_id,
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
        compromised_device = is_fraud()
        compromised_network = is_fraud()

        return {
            "agent_id": self.agent_id,
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

# Define is_fraud() function
def is_fraud(probability=0.05):
    """
    Returns True if a device or network is compromised, based on the probability.
    Default probability of compromise is 5%.
    """
    return random.random() < probability